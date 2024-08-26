import unittest
from helpers.server_type import Server

class serversTest(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        self.server = Server("CPU.S1_low", [1, 60], 15000, 2, 400, 60, 96, 1000, 288, "low")
    
    @classmethod
    def tearDown(self) -> None:
        self.server = None
    
    def test_canBeDeployed(self):
        self.assertTrue(self.server.canBeDeployed(2))

    def test_canNotBeDeployed(self):
        self.assertFalse(self.server.canBeDeployed(61))

    def test_setSellingPrices(self):
        self.server.setSellingPrices(100)
        self.assertEqual(self.server.selling_prices, 100)

    