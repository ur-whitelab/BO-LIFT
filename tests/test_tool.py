from boicl import BOICLTool, Pool
import os


def test_tell_tools():
    pool_list = ["red", "green", "blue"]
    pool = Pool(pool_list)
    tool = BOICLTool(pool)
    with open("test.csv", "w") as f:
        f.write("yellow, 1\n")
        f.write("teal, 2\n")
    tool("Ask")
    s = tool("Tell test.csv")
    assert s == "Added 2 training examples."
    tool("Ask")
    tool("Best")
    os.remove("test.csv")
