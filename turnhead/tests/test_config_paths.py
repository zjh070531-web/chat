import tempfile
import unittest
from pathlib import Path
from src.pipeline import ROOT, read_config, config_snapshot
import yaml


class RelativePathTests(unittest.TestCase):
    def test_project_assets_and_snapshot_reload(self):
        config = read_config(ROOT/'config.yaml')
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            output = Path(directory)
            snapshot = config_snapshot(config, output)
            for section in ('bpjdet', 'head_pose'):
                for key in ('repo', 'weights'):
                    self.assertTrue(config[section][key].is_relative_to(ROOT))
                    self.assertTrue(config[section][key].exists())
                    self.assertFalse(Path(snapshot[section][key]).is_absolute())
            self.assertTrue(config['video'].is_file())
            self.assertTrue(config['video'].is_relative_to(ROOT))
            self.assertFalse(Path(snapshot['video']).is_absolute())
            self.assertFalse(Path(snapshot['output_dir']).is_absolute())
            self.assertTrue(config['standing_filter']['weights'].is_relative_to(ROOT))
            self.assertTrue(config['standing_filter']['weights'].is_file())
            self.assertFalse(Path(snapshot['standing_filter']['weights']).is_absolute())
            if config['standing_filter']['perspective_calibration']:
                self.assertTrue(config['standing_filter']['perspective_calibration'].is_file())
                self.assertFalse(Path(snapshot['standing_filter']['perspective_calibration']).is_absolute())
            path = output/'config.snapshot.yaml'
            path.write_text(yaml.safe_dump(snapshot, allow_unicode=True), encoding='utf-8')
            reloaded = read_config(path)
            self.assertEqual(reloaded, config)


if __name__ == '__main__':
    unittest.main()
