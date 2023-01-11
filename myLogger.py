class myLog:
    def __init__(self, level=0):
        self.level = level

    @staticmethod
    def heading(text):
        print("\n" + 40*"#" + f" {text} ".center(20) + 40*"#")

    @staticmethod
    def subheading(text):
        print("\n" + 20*"-" + f" {text} ".center(20) + 20*"-")

    @staticmethod
    def indent(n, text):
        extras = ''
        for i in range(1, n+1):
            extras += '|' + ' '
        print(extras)
        print(f'{extras}{text}')