import asyncio
import io
import time

from PIL import Image

from steeleagle_mcp.inferred_mission.config import Settings
from steeleagle_mcp.inferred_mission.service import AnalysisService, Runtime


def test_live_sections_queue_in_order_while_model_is_busy(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (128, 64), "blue").save(image, "JPEG")
    jpeg = image.getvalue()
    dsl = """Actions:
    Wait wait(duration = 5.0)
    ReturnToHome home()
Mission:
    Start wait
    During wait:
        done -> home
"""

    async def run_test():
        started = asyncio.Event()
        release = asyncio.Event()
        inferred = []

        async def infer(_prompt, frames):
            inferred.append(frames[0].frame_id)
            if len(inferred) == 1:
                started.set()
                await release.wait()
            return "Wait 5 seconds, then return home."

        async def generate(candidate, _reference):
            assert candidate.dispatch("compile_mission_dsl", {"dsl": dsl})["ok"]

        service = AnalysisService(tmp_path)
        settings = Settings(section_seconds=1, max_frame_side=64)
        run = service.start_live("test1", "local", settings, Runtime({}, infer, generate))
        stamp = time.time() - 4

        def ingest(index):
            service.ingest("test1", "local", stamp + index - 1, jpeg, frame_id=index)

        try:
            ingest(1)
            ingest(2)
            await asyncio.wait_for(started.wait(), 5)
            for index in (3, 4, 5):
                ingest(index)

            snapshot = run.snapshot()
            assert snapshot["collecting_section"] == 5
            assert [row["status"] for row in snapshot["sections"]] == ["inferring", "queued", "queued", "queued"]
            queued_path = run.directory / run.records[2]["frame_files"][0]
            assert queued_path.read_bytes() == jpeg

            service.stop(run)
            assert run.metadata["status"] == "stopping"
            assert run.records[5]["status"] == "queued"
            release.set()
            await asyncio.wait_for(run.task, 10)

            assert inferred == [1, 2, 3, 4, 5]
            assert run.metadata["status"] == "stopped"
            assert all(row["status"] == "completed" for row in run.records.values())
            assert run.records[2]["history"][0]["section"] == 1
            with Image.open(queued_path) as model_input:
                assert model_input.size == (64, 32)
        finally:
            await service.close()

    asyncio.run(run_test())
