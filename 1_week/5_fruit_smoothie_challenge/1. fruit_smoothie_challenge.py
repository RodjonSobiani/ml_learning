import unittest

prices = {
    "Strawberries": "$1.50",
    "Banana": "$0.50",
    "Mango": "$2.50",
    "Blueberries": "$1.00",
    "Raspberries": "$1.00",
    "Apple": "$1.75",
    "Pineapple": "$3.50",
    "___": "___",
}


class Smoothie:
    def __init__(self, ingredients: list):
        self.ingredients = ingredients

    def get_cost(self):
        ingredients = self.ingredients
        sort = list(filter(lambda i: i[0] in ingredients, prices.items()))
        print(sort)

        for cost in sort:
            cost = float(cost[1].replace('$', ''), 2)
            print('COST: ', cost)
            return cost

        # cost = []
        # ingredients = self.ingredients
        # print('1: ', len(ingredients))
        # for i in range(len(ingredients)):
        #     for (key, value) in prices.items():
        #         if key == ' '.join(ingredients):
        #             print('KEY: ', key)
        #             print('VALUE: ', value)
        #             return value
        #         cost.append(value)
        #         print('COST: ', cost)

    def get_name(self):
        ingredients = self.ingredients
        ingredients.sort()
        if len(ingredients) == 1:
            result = ' '.join(str(x.replace('ies', 'y')) for x in ingredients) + ' Smoothie'
        else:
            result = ' '.join(str(x.replace('ies', 'y')) for x in ingredients) + ' Fusion'
        return result


s1 = Smoothie(["Banana"])
s2 = Smoothie(["Raspberries", "Strawberries", "Blueberries"])
s3 = Smoothie(["Mango", "Apple", "Pineapple"])
s4 = Smoothie(["Pineapple", "Strawberries", "Blueberries", "Mango"])
s5 = Smoothie(["Blueberries"])


class TestStringMethods(unittest.TestCase):
    # Test function to test equality of two value
    def test_negative(self):
        self.assertEqual(s1.ingredients, ["Banana"])
        self.assertEqual(s1.get_cost(), "$0.50")
        # self.assertEqual(s1.get_price(), "$1.25")
        self.assertEqual(s1.get_name(), "Banana Smoothie")

        self.assertEqual(s2.ingredients, ["Raspberries", "Strawberries", "Blueberries"])
        self.assertEqual(s2.get_cost(), "$3.50")
        # self.assertEqual(s2.get_price(), "$8.75")
        self.assertEqual(s2.get_name(), "Blueberry Raspberry Strawberry Fusion")

        self.assertEqual(s3.ingredients, ["Mango", "Apple", "Pineapple"])
        self.assertEqual(s3.get_cost(), "$7.75")
        # self.assertEqual(s3.get_price(), "$19.38")
        self.assertEqual(s3.get_name(), "Apple Mango Pineapple Fusion")

        self.assertEqual(s4.ingredients, ["Pineapple", "Strawberries", "Blueberries", "Mango"])
        self.assertEqual(s4.get_cost(), "$8.50")
        # self.assertEqual(s4.get_price(), "$21.25")
        self.assertEqual(s4.get_name(), "Blueberry Mango Pineapple Strawberry Fusion")

        self.assertEqual(s5.ingredients, ["Blueberries"])
        self.assertEqual(s5.get_cost(), "$1.00")
        # self.assertEqual(s5.get_price(), "$2.50")
        self.assertEqual(s5.get_name(), "Blueberry Smoothie")


if __name__ == '__main__':
    unittest.main()
