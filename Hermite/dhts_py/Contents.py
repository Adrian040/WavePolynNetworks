"""Index of the translated Discrete Hermite Transform toolbox."""

def contents():
    return {
        "basic": ["dhtord", "dhtmtx", "samplat", "hermite", "chtmtx", "gbtmtx"],
        "transform": ["dht", "idht", "dht2", "idht2", "dhti", "dhti2", "dht3", "idht3"],
        "multiscale": ["mdht", "imdht", "mdhti", "mdht2", "imdht2", "mdhti2"],
        "quadtree": ["dhtqt", "idhtqt", "im2qtb", "qtb2im", "qtplot"],
        "postprocessing": ["rdht", "rdht2", "ddht", "sdht2", "qdht", "pdht", "xdht2"],
        "binomial": ["fbt", "fbt2", "fbt3", "fbtmtx", "bt2dht", "dht2bt"],
    }

__all__ = ["contents"]
