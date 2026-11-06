# my_app/main.py
from visionai_framework.config import ConfigLoader
from visionai_framework.ingestion import IngestionRequest
from visionai_framework.plugins import PluginManager
from visionai_framework.runtime import FrameworkBootstrap
from my_app.detectors.fire_smoke import FireSmokePlugin

config = ConfigLoader.load("config/vizAi.yaml")
pm = PluginManager()
pm.register_plugin(FireSmokePlugin())
runtime = FrameworkBootstrap.initialize(config=config, plugin_manager=pm).create_sync_runtime()

result = runtime.submit(IngestionRequest(
    source_uri="file:///D:/images/test_fire.jpg",
    source_type="file", modality="image", context_id="site_a", source_id="cam-1",
    start_timestamp="2026-11-06T10:00:00Z", end_timestamp="2026-11-06T10:00:00Z",
))
print(result.status, result.error)
for o in result.outputs:
    print(o.payload)      # {'label': 'fire', 'confidence': 0.87, 'bbox': {...}, ...}
runtime.close()
