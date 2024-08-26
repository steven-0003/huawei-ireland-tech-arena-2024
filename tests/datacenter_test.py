import unittest
from helpers.datacenters import Datacenter
from helpers.server_type import Server

class datacenterTest(unittest.TestCase):
    @classmethod
    def setUp(self) -> None:
        server_types = {"CPU.S1_low": Server("CPU.S1_low", [1, 60], 15000, 2, 400, 60, 96, 1000, 288, "low"),
                        "GPU.S1_low": Server("GPU.S1_low", [1, 72], 120000, 4, 3000, 8, 96, 1000, 2310, "low")}
        self.datacenter = Datacenter("DC1", 0.25, "low", 25245, server_types)
    
    @classmethod
    def tearDown(self) -> None:
        self.datacenter = None

    def test_buyServer(self):
        self.datacenter.buy_server('CPU.S1_low', 'server-1', 1)
        self.assertEqual(self.datacenter.inventory_level, 2)
        self.assertEqual(self.datacenter.inventory['CPU.S1_low'], [(1,'server-1')])

    def test_sellServer(self):
        self.datacenter.inventory['CPU.S1_low'].append((1,'server-1'))
        self.datacenter.inventory_level += 2
        self.datacenter.sell_server('CPU.S1_low')
        self.assertEqual(self.datacenter.inventory_level, 0)
        self.assertEqual(self.datacenter.inventory['CPU.S1_low'], [])

    def test_checkLifetime(self):
        self.datacenter.inventory['CPU.S1_low'].append((1,'server-1'))
        self.datacenter.inventory_level += 2
        self.datacenter.check_lifetime(1)
        self.assertEqual(self.datacenter.inventory['CPU.S1_low'], [(1,'server-1')])
        self.assertEqual(self.datacenter.inventory_level, 2)

    def test_checkLifetimeRemove(self):
        self.datacenter.inventory['CPU.S1_low'].append((1,'server-1'))
        self.datacenter.inventory_level += 2
        self.datacenter.check_lifetime(98)
        self.assertEqual(self.datacenter.inventory_level, 0)
        self.assertEqual(self.datacenter.inventory['CPU.S1_low'], [])

    def test_buyMultipleServers(self):
        self.datacenter.buy_server('CPU.S1_low', 'server-1', 1)
        self.datacenter.buy_server('CPU.S1_low', 'server-2', 1)
        self.datacenter.buy_server('CPU.S1_low', 'server-3', 1)
        self.datacenter.buy_server('GPU.S1_low', 'server-4', 1)
        self.datacenter.buy_server('GPU.S1_low', 'server-5', 1)
        self.datacenter.buy_server('GPU.S1_low', 'server-6', 1)
        self.assertEqual(self.datacenter.inventory_level, 18)
        self.assertEqual(self.datacenter.inventory['CPU.S1_low'], [(1,'server-1'),(1,'server-2'),(1,'server-3')])
        self.assertEqual(self.datacenter.inventory['GPU.S1_low'], [(1,'server-4'),(1,'server-5'),(1,'server-6')])