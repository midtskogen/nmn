import unittest
import json
from unittest.mock import patch, mock_open
import os
import sys
from datetime import datetime, timezone

# Add the parent directory to the Python path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import the module we want to test
from shared_utils import atomic_json_rw, update_quota_tracker

class TestSharedUtils(unittest.TestCase):

    def test_atomic_json_rw(self):
        """
        Tests the atomic_json_rw context manager for reading and writing JSON data.
        """
        initial_json_data = '{"key": "initial_value"}'
        
        # We use mock_open to simulate the file without actually touching the disk.
        m = mock_open(read_data=initial_json_data)

        # We patch 'open', 'fcntl.flock', and 'os.path.exists' to simulate file presence.
        with patch('builtins.open', m), \
             patch('fcntl.flock'), \
             patch('os.path.exists', return_value=True): # <-- FIX: This line was added

            with atomic_json_rw('fake_path.json') as data:
                # 1. Verify that the initial data was read correctly
                self.assertEqual(data, {"key": "initial_value"})
                
                # 2. Modify the data
                data['new_key'] = 'new_value'
            
            # 3. Verify that the file was opened for writing
            m.assert_called_with('fake_path.json', 'w')
            
            # 4. Verify that the modified data was written to the file
            handle = m()
            written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
            self.assertEqual(json.loads(written_data), {"key": "initial_value", "new_key": "new_value"})

    def test_update_quota_tracker(self):
        """
        Tests the quota tracking logic.
        """
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        initial_tracker_data = {
            today_str: {}
        }
        
        updates_to_apply = {
            "ams1": 1024 * 1024 * 5,  # 5 MB
            "ams2": 1024 * 1024 * 10  # 10 MB
        }
        user_ip = "192.168.1.100"
        task_id = "test_task_123"
        fake_quota_file_path = "/fake/path/quota_tracker.json"
        
        mock_data = initial_tracker_data.copy()

        with patch('shared_utils.atomic_json_rw') as mock_atomic_rw:
            mock_atomic_rw.return_value.__enter__.return_value = mock_data
            update_quota_tracker(updates_to_apply, task_id, user_ip, fake_quota_file_path)

            station1_usage = mock_data[today_str]["ams1"]
            station2_usage = mock_data[today_str]["ams2"]

            self.assertEqual(station1_usage["total"], 5242880)
            self.assertEqual(station1_usage["sites"][user_ip], 5242880)
            self.assertEqual(station2_usage["total"], 10485760)
            self.assertEqual(station2_usage["sites"][user_ip], 10485760)

if __name__ == '__main__':
    unittest.main()

