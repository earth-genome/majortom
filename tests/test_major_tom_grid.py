import random
import unittest
import shapely.geometry
from shapely.geometry import Polygon, Point
from src.majortom_eg.MajorTom import MajorTomGrid,GridCell

# geometries for testing
bigSouthampton = {
    "coordinates": [
        [
            [
                -76.35673421721803,
                39.55614384974018
            ],
            [
                -76.35673421721803,
                39.53123810591927
            ],
            [
                -76.3131967920373,
                39.53123810591927
            ],
            [
                -76.3131967920373,
                39.55614384974018
            ],
            [
                -76.35673421721803,
                39.55614384974018
            ]
        ]
    ],
    "type": "Polygon"
}
southampton =  {
                "coordinates": [
                    [
                        [
                            -76.34311254543559,
                            39.548984543610345
                        ],
                        [
                            -76.34311254543559,
                            39.53895755383081
                        ],
                        [
                            -76.3260548926393,
                            39.53895755383081
                        ],
                        [
                            -76.3260548926393,
                            39.548984543610345
                        ],
                        [
                            -76.34311254543559,
                            39.548984543610345
                        ]
                    ]
                ],
                "type": "Polygon"
            }
bsIds = [
"dr19n8zgg4e",
"dr18zj1ntew",
"dr18zjjnxgw",
"dr18zkfwjxt",
"dr18yt6wvpt",
"dr19p2j4syu",
"dr18yqy0cg1",
"dr18zr195x5",
"dr19p2dg7nx",
"dr19n8p6tyv",
"dr18yvbx8p6",
"dr19n3mksf8",
"dr18zrh34rp",
"dr19nbv7bfe",
"dr19p2ygvd9",
"dr18yymbjg3",
"dr18zqkb5e2",
"dr18ys9vfgf",
"dr19n86504v",
"dr19p0feyde",
"dr18zmypwzm",
"dr18yqw03z3",
"dr18ywz8cg4",
"dr19n8y7zdt",
"dr18zhwjy5y",
"dr18ytdxm5t",
"dr19nbxgrwx",
"dr18zrf42ff",
"dr18ywbbbej",
"dr18ykrj7zz",
"dr18zhvnpzw",
"dr18zjdzq79",
"dr18yy3b57r",
"dr18zpbd2fc",
"dr18znr2jg3",
"dr18yzu6q6z",
"dr19p92sw4w",
"dr18yx5chxn",
"dr18yyf2bej",
"dr18yt0ysed",
"dr19n8775dg",
"dr19nbkg5dv",
"dr18yxffq6y",
"dr18zrbdm6u",
"dr18yzh11r1",
"dr18yyw26zr",
"dr18zhxtceg",
"dr18zsdmv5b",
"dr18zmnwxgw",
"dr18yv6ycxx",
"dr18yt5we7w",
"dr18zmjqse8",
"dr19p12ksft",
"dr18yrn3nzp",
"dr18yx31tg4",
"dr18yzk1971",
"dr18yzqc85j",
"dr18zny2ze1",
"dr18yvtx2ee",
"dr19n8q546z",
"dr19n8r7jfv",
"dr19p02gjfz",
"dr19p2rg5fb",
"dr18zrx3yx1",
"dr19p2354dg",
"dr18zq60hek",
"dr18zp13pp0",
"dr18yye8kp7",
"dr18yusty5z",
"dr18yufqhxd",
"dr18yv2nfzs",
"dr19n2r716c",
"dr18yxs3czh",
"dr18zp9cupp",
"dr19n3jkhy8",
"dr19n92heds",
"dr18ytnqd79",
"dr18zjbxsxk",
"dr19n9js1qt",
"dr18yzx1vr5",
"dr18zhemz7g",
"dr18zm1qd5x",
"dr18ywmb15q",
"dr18zmtz3ex",
"dr18zrg6k6b",
"dr18yvmwbxe",
"dr18zj2nzrw",
"dr19p37k86w",
"dr18zp33x50",
"dr18zph1jzh",
"dr18zr39ee5",
"dr18yuwjceu",
"dr18zkxty5v",
"dr18zt8x2ed",
"dr18yzvf3fv",
"dr18zw2bjg3",
"dr18yz99zx5",
"dr18yw8b2xm",
"dr18znc0cg4",
"dr18yvjne7t",
"dr18zt0ne7e",
"dr18zjsr3ex",
"dr18zn725e2",
"dr18yxn95x4",
"dr19n8d7mqe",
"dr18ymvxsxk",
"dr18zn28hek",
"dr19p267n4b",
"dr19p246wnb",
"dr18ywf0u55",
"dr18yu3vrru",
"dr18yuktqpz",
"dr18zhkv6zb",
"dr18yxcd74b",
"dr18yxvdm6f",
"dr18zm8rk58",
"dr18zrd1bz4",
"dr18zrk3d7p",
"dr18yyz8y7h",
"dr18zqc0y7j",
"dr18yttrk59",
"dr19p0t7mqx",
"dr18yw2257r",
"dr18zq78056",
"dr18yxu63fu",
"dr19n2zey6x",
"dr18zngbben",
"dr18ys8mz7v",
"dr18ymxzq7d",
"dr19n8hdxnc",
"dr19p0469qv",
"dr18yzb6k4v",
"dr18zpcfk4z",
"dr18ywk2jg3",
"dr18zq2b15q",
"dr18zn903z6",
"dr18zqebmr2",
"dr18yv1n95s",
"dr18ytwxr7w",
"dr19p0u5f49",
"dr19p285kqe",
"dr18zmxpm5s",
"dr18zpud74f",
"dr19nc2k869",
"dr18yww0qrk",
"dr18ysejy7b",
"dr19nbq5p4g",
"dr18yzs3yrp",
"dr19nc4upnt",
"dr18znf2v70",
"dr18zkkvrrc",
"dr18yt9r3g8",
"dr19n82ehdz",
"dr18zrs9zxn",
"dr19p1muddt",
"dr19p30s1wd",
"dr18ysfq0p8",
"dr19p2te6ne",
"dr19nbpddwc",
"dr19p0j48qc",
"dr18yx69970",
"dr18yx11jz4",
"dr18yzm3sep",
"dr18ywr0hek",
"dr18yt4q858",
"dr18yv5wxgx",
"dr18yvvztz3",
"dr18zn60056",
"dr19p0rep4v",
"dr19p2qe06f",
"dr18zp21ee5",
"dr18yqz2v5p",
"dr18zkrtqpv",
"dr19p0v7v6x",
"dr18yqr0056",
"dr18zrz6qdc",
"dr19p10khyt",
"dr19p32s9dd",
"dr18zkdmb7y",
"dr18ympw95t",
"dr18zj4w95e",
"dr18ytmqup9",
"dr19p35k0qw",
"dr18zrp11r5",
"dr18zx3c85n",
"dr18yybbv70",
"dr19nbfecfx",
"dr18yuzy4r8",
"dr18ytbrerr",
"dr18yzz4m6g",
"dr19p0deqwe",
"dr18yvdz3ex",
"dr19p2u5ydt",
"dr18yvgpwzm",
"dr18zm2qup8",
"dr19p15h5y9",
"dr18yxm385h",
"dr18ysrm2xc",
"dr19n2p69qc",
"dr18yznc0pj",
"dr18ykxjggz",
"dr18yu9vz7u",
"dr18zk8vv5c",
"dr18zhrt3xg",
"dr18yv4qsed",
"dr18zmhn95d",
"dr18ytsp2ee",
"dr18yyy2fgp",
"dr18zneb2xq",
"dr18zqv2ben",
"dr18zwc2g50",
"dr18zk6m2ry",
"dr18zjnwe7s",
"dr18znk8ngm",
"dr18zmsrq79",
"dr18zpvfqdb",
"dr19n8xg7ne",
"dr18yy28056",
"dr19nb7e04z",
"dr19p06716v",
"dr19n90h5ws",
"dr18zq90qrm",
"dr18yuxmv5f",
"dr19p8b7bfx",
"dr18yzg474c",
"dr18zpg4rdu",
"dr18ywubfgn",
"dr18znzbfgp",
"dr18zhsvfgb",
"dr18yryfk4u",
"dr18zqq0ngm",
"dr19n8t5knx",
"dr19nbc5yd9",
"dr18yu8tggy",
"dr18yuemfgc",
"dr18zhdjggv",
"dr18zhfw1pe",
"dr18ys3v6zf",
"dr18zksvz7c",
"dr18zjtxm5t",
"dr19nb44syc",
"dr19p0q75dz",
"dr18yxrcx50",
"dr18zpm9974",
"dr18zqy8u5h",
"dr18zketggu",
"dr18ykzw1pt",
"dr18yw38ngm",
"dr18zj6yyr9",
"dr19p07ehfg",
"dr19n96udd9",
"dr18zj5yse9",
"dr18yv9rq7d",
"dr18yvspm5t",
"dr18yvwzk58",
"dr19p2b5u6e",
"dr19p0pdxnv",
"dr18ystvv5g",
"dr18ytgpdr7",
"dr19p27g16y",
"dr18znq0477",
"dr19ncjsnn9",
"dr18yy8bmr2",
"dr18zh6j7zv",
"dr19nbt72ye",
"dr18zpz63fy",
"dr19p17hef9",
"dr19p3mux6d",
"dr19nbre4dc",
"dr18zmgx8p6",
"dr18zm9xr7w",
"dr18zm7nfze",
"dr18yz33d7j",
"dr18zr6cx51",
"dr19n8ggbf9",
"dr19p0c7g4t",
"dr19p0m506c",
"dr18zjypdr7",
"dr18ytpwtew",
"dr18zjpywgd",
"dr18yrrcd7j",
"dr19n8f7v6e",
"dr18yyg8u55",
"dr19nbde3yx",
"dr19nbyeu69",
"dr18zpx3czn",
"dr19p2m5hfu",
"dr18zqr8477",
"dr18zpk1tgh",
"dr18ywg2ze0",
"dr18ysxmbec",
"dr19nbzgzdx",
"dr19n8448nv",
"dr18yyk8477",
"dr18yzcdrdg",
"dr19nc0k0q9",
"dr18znv0u5h",
"dr18yurmmpf",
"dr18yszwjxw",
"dr19p2fgg4x",
"dr19n9ms96t",
"dr18yz69tg5",
"dr18zjgrex2",
"dr18yx7csen",
"dr18yz134rj",
"dr18zr2385n",
"dr18zm4wtet",
"dr19p2vef4e",
"dr18yrq3wgp",
"dr18zpnchz0",
"dr19nb2g16g",
"dr18ytvz9pq",
"dr19nb65hfc",
"dr19p1ju4wt",
"dr18zpqcsg0",
"dr18zrudrdy",
"dr18yvpyd78",
"dr18yv8p6gs",
"dr18yyr215q",
"dr18zps9gp4",
"dr18zxf4m6v",
"dr18ywe2rx2",
"dr18yyubze1",
"dr19n8v5u4x",
"dr18zj9x6gs",
"dr18yy70ngm",
"dr18yxx1bxp",
"dr19nc6ux4t",
"dr19ncmsw49",
"dr18yw70477",
"dr18yx99gp0",
"dr18zp6cd7n",
"dr18ywy0y7h",
"dr18zqw8kpk",
"dr18zhzynzd",
"dr18zkvqhx8",
"dr19nbhfewv",
"dr19n94u4w9",
"dr18zqgbv70",
"dr18ysstcey",
"dr18ythywge",
"dr18zmep6ge",
"dr18yrwcuph",
"dr18zh8vb7z",
"dr18ysuynze",
"dr18yvnqwgd",
"dr18ymryyrd",
"dr19p2c7zf9",
"dr18zjxp2ed",
"dr18zrr1975",
"dr18zjmwvpt",
"dr18zry474u",
"dr18yxq9ee4",
"dr18zqf8cg5",
"dr19p0ygbft",
"dr19n8kep4c",
"dr18zmbz9pq",
"dr18yxb4rdf",
"dr18znw2rx3",
"dr18zkwmfez",
"dr18zj8pr7w",
"dr18zrm9tgj",
"dr19n8c5cft",
"dr19n2xeqqx",
"dr18yqm8hek",
"dr19p3jupqd",
"dr18yskt3xy",
"dr19nbggv6t",
"dr18yuvn5rt",
"dr18zmmycxx",
"dr18yxz42dz",
         ]

class TestGridCell(unittest.TestCase):

    def test_grid_cell_init_and_id(self):
        # Create a simple square polygon around (0,0)
        poly = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        cell = GridCell(poly)
        self.assertIsInstance(cell.geom, Polygon, "GridCell should store a Polygon geometry")

        # Test the ID method - Just ensure it returns a string of length 11
        cell_id = cell.id()
        self.assertIsInstance(cell_id, str, "GridCell id() should return a string")
        self.assertEqual(len(cell_id), 11, "GridCell geohash ID should be length 11")


class TestMajorTomGrid(unittest.TestCase):

    def setUp(self):
        # Default MajorTomGrid with D=320 and overlap=True
        self.grid = MajorTomGrid(d=320, overlap=True)

    def test_generate_grid_cells_basic(self):
        # Create a simple polygon around the equator (e.g., a 2x2 degree box)
        test_poly = shapely.geometry.shape(bigSouthampton)

        cells = list(self.grid.generate_grid_cells(test_poly))
        # There should be at least one cell intersecting this polygon
        self.assertTrue(len(cells) == 335, "Should generate 117 cells")
        # Check that all returned cells intersect the original polygon
        for cell in cells:
            self.assertTrue(cell.geom.intersects(test_poly), "All returned cells should intersect the polygon")

    def test_generate_grid_cells_overlap_off(self):
        # Test with overlap turned off
        grid_no_overlap = MajorTomGrid(d=320, overlap=False)
        test_poly = shapely.geometry.shape(bigSouthampton)

        cells = list(grid_no_overlap.generate_grid_cells(test_poly))
        # Ensure we have fewer cells when overlap is off compared to when it's on
        cells_with_overlap = list(self.grid.generate_grid_cells(test_poly))
        self.assertTrue(len(cells) < len(cells_with_overlap),
                        "Number of cells generated should be smaller when overlap is False")

    def test_small_polygon(self):
        # Test a very small polygon around a point
        # This is to ensure that even a tiny polygon yields some cells if it intersects a grid cell
        small_poly = Point(0, 0).buffer(0.0001)  # Very small circle around (0,0)
        cells = list(self.grid.generate_grid_cells(small_poly))
        self.assertTrue(len(cells) > 0, "A very small polygon at the equator should still yield at least one cell")

    def test_polygon_outside_area(self):
        # Test a polygon that does not intersect any grid cells (e.g., one far outside normal lat ranges)
        # Actually, latitudes don't go beyond -90 and 90, so let's just pick a polygon well away from 0,0 and see
        test_poly = Polygon([
            (170, 80),
            (171, 80),
            (171, 81),
            (170, 81)
        ])
        cells = list(self.grid.generate_grid_cells(test_poly))
        # Even at high latitudes, should get some cells if it intersects (80,81) is valid
        self.assertTrue(len(cells) > 0, "High-latitude polygon should yield cells if it falls within valid lat range")

    def test_cell_lookup(self):
        cells = list(self.grid.generate_grid_cells(shapely.geometry.shape(bigSouthampton)))
        cell = random.choice(cells)
        found_cell = self.grid.cell_from_id(cell.id())
        self.assertTrue(found_cell is not None,)
        self.assertTrue(found_cell.geom.equals_exact(cell.geom,0.00000001))


if __name__ == '__main__':
    unittest.main()
