import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nci import make_commands


class StageFourCommandTests(unittest.TestCase):
    def test_stage_four_has_one_safe_command_per_dataset_and_classifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'nci').mkdir()
            (root / 'nci' / 'run_task.sh').write_text(
                '#!/bin/bash\n', encoding='utf-8')

            counts = {'msc_new': 23, 'tox21.data': 12, 'HTSFP10': 2}
            for folder, count in counts.items():
                directory = root / folder
                directory.mkdir()
                for i in range(count):
                    stem = f'dataset_{folder.replace(".", "_")}_{i}'
                    (directory / f'{stem}.tr_0.2_1.txt').write_text(
                        'train\n', encoding='utf-8')
                    (directory / f'{stem}.te_0.2_1.txt').write_text(
                        'test\n', encoding='utf-8')

            old_root = make_commands.ROOT
            make_commands.ROOT = root
            argv = [
                'make_commands.py',
                '--groups', 'ED', 'tox21', 'HTSFP10',
                '--tune-others', '50',
                '--dims', '1000',
                '--node-features', '18',
                '--features', 'concat',
                '--clfs', 'rf', 'lr', 'xgb',
                '--rdkit', '0',
                '--results-dir', 'results_tuned_dim1000',
                '--out-dir', 'cmds',
                '--python', 'python',
            ]
            try:
                with mock.patch.object(sys, 'argv', argv):
                    make_commands.main()
            finally:
                make_commands.ROOT = old_root

            commands = root / 'cmds'
            stage1 = (commands / 'stage1_prepare.txt').read_text(
                encoding='utf-8').splitlines()
            stage2 = (commands / 'stage2_embed.txt').read_text(
                encoding='utf-8').splitlines()
            stage3 = (commands / 'stage3_runs.txt').read_text(
                encoding='utf-8').splitlines()
            stage4 = (commands / 'stage4_tuning.txt').read_text(
                encoding='utf-8').splitlines()

            self.assertEqual(len(stage1), 37)
            self.assertEqual(len(stage2), 37)
            self.assertEqual(len(stage3), 111)
            self.assertEqual(len(stage4), 111)
            self.assertTrue(all(
                ' Run_other_classifier_tuning.py -i ' in line
                for line in stage4))
            self.assertTrue(all(
                ' Run_other_classifier_tuning.py ' not in line
                for line in stage1 + stage2 + stage3))
            self.assertTrue(all(
                not any(char in line for char in ['"', "'", '&&', ' > '])
                for line in stage4))


if __name__ == '__main__':
    unittest.main()
