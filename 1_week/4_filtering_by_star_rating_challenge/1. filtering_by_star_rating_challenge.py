import unittest


def filter_by_rating(d, rating):
    new_dict = dict()
    for (key, value) in d.items():
        if value == rating:
            new_dict[key] = value
    if len(new_dict) != 0:
        return new_dict
    else:
        return 'No results found'


filter_by_rating({
    "Luxury Chocolates": "*****",
    "Tasty Chocolates": "****",
    "Aunty May Chocolates": "*****",
    "Generic Chocolates": "***"
}, "*****")

filter_by_rating({
    "Continental Hotel": "****",
    "Big Street Hotel": "**",
    "Corner Hotel": "**",
    "Trashviews Hotel": "*",
    "Hazbins": "*****"
}, "*")

filter_by_rating({
    "Solo Restaurant": "***",
    "Finest Dinings": "*****",
    "Burger Stand": "***"
}, "****")


class TestStringMethods(unittest.TestCase):
    # Test function to test equality of two value
    def test_negative(self):
        self.assertEqual(filter_by_rating(
            {"Brand A": "***", "Brand B": "***"}, "*****"),
            "No results found"
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "*****", "Brand C": "*",
             "Brand D": "**", "Brand E": "****", "Brand F": "*****",
             "Brand G": "****", "Brand H": "****", "Brand I": "*****",
             "Brand K": "***", "Brand L": "*****", "Brand M": "***",
             "Brand N": "*", "Brand O": "***", "Brand P": "*****",
             "Brand Q": "**", "Brand R": "****"}, "***"),
            {"Brand K": "***", "Brand M": "***", "Brand O": "***"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "***", "Brand C": "**",
             "Brand D": "*****", "Brand E": "*", "Brand F": "****",
             "Brand G": "*****", "Brand H": "*****", "Brand I": "**",
             "Brand K": "*", "Brand L": "*", "Brand M": "***",
             "Brand N": "*", "Brand O": "*", "Brand P": "**",
             "Brand Q": "**", "Brand R": "****", "Brand S": "****",
             "Brand T": "**", "Brand U": "*", "Brand V": "*",
             "Brand W": "*", "Brand X": "***", "Brand Y": "*****",
             "Brand Z": "****"}, "**"),
            {"Brand C": "**", "Brand I": "**", "Brand P": "**",
             "Brand Q": "**", "Brand T": "**"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "***", "Brand B": "**", "Brand C": "****",
             "Brand D": "*", "Brand E": "*", "Brand F": "**",
             "Brand G": "***", "Brand H": "*", "Brand I": "**",
             "Brand K": "*****", "Brand L": "**", "Brand M": "*"}, "**"),
            {"Brand B": "**", "Brand F": "**", "Brand I": "**", "Brand L": "**"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "***", "Brand C": "***",
             "Brand D": "***", "Brand E": "*", "Brand F": "**",
             "Brand G": "***", "Brand H": "*****", "Brand I": "**",
             "Brand K": "***", "Brand L": "*", "Brand M": "****",
             "Brand N": "****", "Brand O": "***", "Brand P": "**",
             "Brand Q": "*****", "Brand R": "*", "Brand S": "*",
             "Brand T": "*****", "Brand U": "*****", "Brand V": "*",
             "Brand W": "*****", "Brand X": "****",
             "Brand Y": "*", "Brand Z": "*****"}, "****"),
            {"Brand M": "****", "Brand N": "****", "Brand X": "****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "****", "Brand C": "*****",
             "Brand D": "*", "Brand E": "**", "Brand F": "***",
             "Brand G": "*", "Brand H": "**", "Brand I": "*",
             "Brand K": "**", "Brand L": "****"}, "*"),
            {"Brand A": "*", "Brand D": "*", "Brand G": "*", "Brand I": "*"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "****", "Brand B": "****", "Brand C": "**",
             "Brand D": "*", "Brand E": "**", "Brand F": "***",
             "Brand G": "***", "Brand H": "**", "Brand I": "*",
             "Brand K": "*", "Brand L": "****", "Brand M": "*",
             "Brand N": "*****", "Brand O": "**", "Brand P": "*",
             "Brand Q": "*****", "Brand R": "*"}, "****"),
            {"Brand A": "****", "Brand B": "****", "Brand L": "****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "**", "Brand B": "*", "Brand C": "*"}, "**"),
            {"Brand A": "**"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "****", "Brand B": "*", "Brand C": "****",
             "Brand D": "***", "Brand E": "*****"}, "**"),
            "No results found"
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "****", "Brand B": "****", "Brand C": "***",
             "Brand D": "****", "Brand E": "*****",
             "Brand F": "*", "Brand G": "****", "Brand H": "*****",
             "Brand I": "*", "Brand K": "****",
             "Brand L": "****", "Brand M": "*", "Brand N": "***",
             "Brand O": "**", "Brand P": "*", "Brand Q": "*",
             "Brand R": "****", "Brand S": "*****", "Brand T": "****",
             "Brand U": "*****", "Brand V": "****",
             "Brand W": "****", "Brand X": "**", "Brand Y": "*"}, "****"),
            {"Brand A": "****", "Brand B": "****", "Brand D": "****", "Brand G": "****",
             "Brand K": "****", "Brand L": "****", "Brand R": "****", "Brand T": "****",
             "Brand V": "****", "Brand W": "****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "**", "Brand B": "****", "Brand C": "***",
             "Brand D": "****", "Brand E": "*", "Brand F": "*",
             "Brand G": "**", "Brand H": "***", "Brand I": "***",
             "Brand K": "**", "Brand L": "***", "Brand M": "**",
             "Brand N": "**", "Brand O": "*", "Brand P": "*",
             "Brand Q": "*****", "Brand R": "***", "Brand S": "**",
             "Brand T": "*", "Brand U": "**", "Brand V": "*",
             "Brand W": "**", "Brand X": "****"}, "**"),
            {"Brand A": "**", "Brand G": "**", "Brand K": "**",
             "Brand M": "**", "Brand N": "**",
             "Brand S": "**", "Brand U": "**", "Brand W": "**"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "**", "Brand C": "****",
             "Brand D": "*****", "Brand E": "*****",
             "Brand F": "*****", "Brand G": "****", "Brand H": "*",
             "Brand I": "*", "Brand K": "*", "Brand L": "****",
             "Brand M": "*", "Brand N": "***", "Brand O": "****",
             "Brand P": "****", "Brand Q": "****",
             "Brand R": "****", "Brand S": "**", "Brand T": "****"}, "*"),
            {"Brand A": "*", "Brand H": "*", "Brand I": "*",
             "Brand K": "*", "Brand M": "*"}
        )
        self.assertEqual(
            filter_by_rating(
                {"Brand A": "**", "Brand B": "*", "Brand C": "*****",
                 "Brand D": "*****"}, "*****"),
            {"Brand C": "*****", "Brand D": "*****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*****", "Brand B": "***", "Brand C": "***",
             "Brand D": "***", "Brand E": "***",
             "Brand F": "***", "Brand G": "****", "Brand H": "*",
             "Brand I": "**", "Brand K": "***", "Brand L": "****",
             "Brand M": "*", "Brand N": "*****", "Brand O": "**",
             "Brand P": "*", "Brand Q": "****", "Brand R": "**",
             "Brand S": "****", "Brand T": "*", "Brand U": "*****",
             "Brand V": "**", "Brand W": "*", "Brand X": "**",
             "Brand Y": "*****"}, "****"),
            {"Brand G": "****", "Brand L": "****", "Brand Q": "****", "Brand S": "****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "*****", "Brand C": "*****",
             "Brand D": "*", "Brand E": "****", "Brand F": "*",
             "Brand G": "****", "Brand H": "*****", "Brand I": "***",
             "Brand K": "***", "Brand L": "***",
             "Brand M": "*", "Brand N": "****", "Brand O": "****",
             "Brand P": "**", "Brand Q": "*****",
             "Brand R": "***"}, "****"),
            {"Brand E": "****", "Brand G": "****", "Brand N": "****", "Brand O": "****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "***", "Brand B": "****", "Brand C": "****",
             "Brand D": "*", "Brand E": "**", "Brand F": "****",
             "Brand G": "*****", "Brand H": "****", "Brand I": "*"}, "****"),
            {"Brand B": "****", "Brand C": "****", "Brand F": "****", "Brand H": "****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "*****", "Brand C": "**",
             "Brand D": "*****", "Brand E": "**", "Brand F": "*",
             "Brand G": "**", "Brand H": "***", "Brand I": "***",
             "Brand K": "*****"}, "*****"),
            {"Brand B": "*****", "Brand D": "*****", "Brand K": "*****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "****", "Brand B": "****", "Brand C": "*****",
             "Brand D": "*****", "Brand E": "****", "Brand F": "***",
             "Brand G": "**", "Brand H": "**", "Brand I": "****",
             "Brand K": "****", "Brand L": "****", "Brand M": "****",
             "Brand N": "***", "Brand O": "**"}, "****"),
            {"Brand A": "****", "Brand B": "****", "Brand E": "****", "Brand I": "****",
             "Brand K": "****", "Brand L": "****", "Brand M": "****"}
        )
        self.assertEqual(
            filter_by_rating({"Brand A": "***", "Brand B": "***"}, "*****"),
            "No results found"
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "***", "Brand B": "*****", "Brand C": "*",
             "Brand D": "****", "Brand E": "*", "Brand F": "**",
             "Brand G": "**", "Brand H": "*****", "Brand I": "**",
             "Brand K": "****", "Brand L": "**", "Brand M": "**",
             "Brand N": "****", "Brand O": "****", "Brand P": "*****"}, "***"),
            {"Brand A": "***"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "**", "Brand B": "*", "Brand C": "*****",
             "Brand D": "*****", "Brand E": "*", "Brand F": "***",
             "Brand G": "*", "Brand H": "**", "Brand I": "*",
             "Brand K": "**", "Brand L": "*", "Brand M": "***",
             "Brand N": "*****", "Brand O": "*"}, "*****"),
            {"Brand C": "*****", "Brand D": "*****", "Brand N": "*****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "*", "Brand C": "*",
             "Brand D": "***", "Brand E": "****", "Brand F": "***",
             "Brand G": "*****", "Brand H": "**", "Brand I": "*",
             "Brand K": "*****", "Brand L": "***",
             "Brand M": "***", "Brand N": "***", "Brand O": "**",
             "Brand P": "**", "Brand Q": "*****",
             "Brand R": "****", "Brand S": "***", "Brand T": "****",
             "Brand U": "*****", "Brand V": "***",
             "Brand W": "*****", "Brand X": "*****", "Brand Y": "***"}, "*****"),
            {"Brand G": "*****", "Brand K": "*****", "Brand Q": "*****",
             "Brand U": "*****", "Brand W": "*****", "Brand X": "*****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*****", "Brand B": "****", "Brand C": "****",
             "Brand D": "*", "Brand E": "*",
             "Brand F": "****", "Brand G": "****", "Brand H": "**",
             "Brand I": "****", "Brand K": "****",
             "Brand L": "*****", "Brand M": "*****", "Brand N": "***",
             "Brand O": "****", "Brand P": "**",
             "Brand Q": "***", "Brand R": "***", "Brand S": "*****",
             "Brand T": "*", "Brand U": "*****",
             "Brand V": "****", "Brand W": "***"}, "****"),
            {"Brand B": "****", "Brand C": "****", "Brand F": "****", "Brand G": "****",
             "Brand I": "****", "Brand K": "****", "Brand O": "****", "Brand V": "****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "****", "Brand C": "*",
             "Brand D": "*****", "Brand E": "**", "Brand F": "****",
             "Brand G": "***", "Brand H": "****", "Brand I": "*",
             "Brand K": "*", "Brand L": "*****",
             "Brand M": "*****", "Brand N": "*", "Brand O": "**",
             "Brand P": "*****", "Brand Q": "**",
             "Brand R": "*****", "Brand S": "*****", "Brand T": "****",
             "Brand U": "*****", "Brand V": "*****",
             "Brand W": "**", "Brand X": "***"}, "**"),
            {"Brand E": "**", "Brand O": "**", "Brand Q": "**", "Brand W": "**"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "**", "Brand B": "**", "Brand C": "**",
             "Brand D": "***", "Brand E": "*****", "Brand F": "**"},
            "****"),
            "No results found"
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "*", "Brand B": "*", "Brand C": "**",
             "Brand D": "*", "Brand E": "****", "Brand F": "****",
             "Brand G": "**", "Brand H": "*", "Brand I": "***",
             "Brand K": "**", "Brand L": "***", "Brand M": "***",
             "Brand N": "****", "Brand O": "*", "Brand P": "*****",
             "Brand Q": "*****", "Brand R": "*",
             "Brand S": "****", "Brand T": "****", "Brand U": "*",
             "Brand V": "**", "Brand W": "****",
             "Brand X": "****", "Brand Y": "****", "Brand Z": "**"}, "***"),
            {"Brand I": "***", "Brand L": "***", "Brand M": "***"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "**", "Brand B": "*****", "Brand C": "***",
             "Brand D": "**", "Brand E": "*", "Brand F": "****",
             "Brand G": "****", "Brand H": "*", "Brand I": "*",
             "Brand K": "*", "Brand L": "****", "Brand M": "*",
             "Brand N": "**", "Brand O": "*", "Brand P": "**", "Brand Q": "*"}, "*****"),
            {"Brand B": "*****"}
        )
        self.assertEqual(filter_by_rating(
            {"Brand A": "****", "Brand B": "*****", "Brand C": "*****",
             "Brand D": "****", "Brand E": "**",
             "Brand F": "*", "Brand G": "**", "Brand H": "**",
             "Brand I": "***", "Brand K": "***", "Brand L": "***",
             "Brand M": "***", "Brand N": "****", "Brand O": "*****",
             "Brand P": "*", "Brand Q": "*", "Brand R": "****",
             "Brand S": "**", "Brand T": "**", "Brand U": "*****",
             "Brand V": "***", "Brand W": "***"}, "**"),
            {"Brand E": "**", "Brand G": "**", "Brand H": "**", "Brand S": "**", "Brand T": "**"}
        )


if __name__ == '__main__':
    unittest.main()
