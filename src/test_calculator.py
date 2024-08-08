import json
import unittest
from unittest.mock import patch, ANY, AsyncMock
import requests
from datetime import datetime
from dataService import send_chat_history_to_springboot, handle_connection, main  # 替换为你的模块名称


class TestSendChatHistory(unittest.TestCase):

    @patch('dataService.requests.post')  # 替换为你的模块名称
    def test_send_chat_history_success(self, mock_post):
        # 模拟请求返回状态码为201
        mock_post.return_value.status_code = 201

        # 输入数据
        user_id = 1
        user_message = "Hello"
        bot_message = "Hi there"

        # 调用被测试的函数
        send_chat_history_to_springboot(user_id, user_message, bot_message)

        # 验证 mock_post 的调用，并忽略具体的 timestamp 值
        mock_post.assert_called_once_with(
            'http://localhost:8080/chathistories',
            json={
                'userMessage': user_message,
                'botMessage': bot_message,
                'timestamp': ANY,  # 使用 ANY 忽略 timestamp 的具体值
                'user_id': user_id
            },
            headers={'Content-Type': 'application/json'}
        )

    @patch('dataService.requests.post')  # 替换为你的模块名称
    def test_send_chat_history_failure(self, mock_post):
        # 模拟请求返回状态码为500
        mock_post.return_value.status_code = 500

        # 输入数据
        user_id = 1
        user_message = "Hello"
        bot_message = "Hi there"

        # 捕获日志输出
        with self.assertLogs(level='ERROR'):
            send_chat_history_to_springboot(user_id, user_message, bot_message)

if __name__ == "__main__":
    unittest.main()
