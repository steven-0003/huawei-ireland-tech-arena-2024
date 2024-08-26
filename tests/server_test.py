import unittest
from helpers.server_type import Server

class serversTest(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        self.server = Server("CPU.S1", [1, 60], 15000, 2, 400, 60, 96, 1000, 288)
    
    @classmethod
    def tearDown(self) -> None:
        self.server = None
    
    def test_canBeDeployed(self):
        self.assertTrue(self.server.canBeDeployed(2))

    def test_canNotBeDeployed(self):
        self.assertFalse(self.server.canBeDeployed(61))


    ## selling_prices for a server are now a dictionary which maps the latency_sensitivity to its price
    ## e.g. selling_prices["low"] = 10
    ## e.g. selling_prices["high"] = 40 
    def test_setSellingPrice(self):

        selling_prices = {
            "low" : 10,
            "medium" : 20,
            "high" : 30
        }

        self.server.setSellingPrices(selling_prices)
        self.assertEqual(self.server.selling_prices, selling_prices)

    