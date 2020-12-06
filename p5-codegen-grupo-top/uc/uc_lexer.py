import argparse
import pathlib
import sys
import ply.lex as lex
import re

class UCLexer:
    """A lexer for the uC language. After building it, set the
    input text with input(), and call token() to get new
    tokens.
    """

    def __init__(self, error_func):
        """Create a new Lexer.
        An error function. Will be called with an error
        message, line and column as arguments, in case of
        an error during lexing.
        """
        self.error_func = error_func
        self.filename = ""

        # Keeps track of the last token returned from self.token()
        self.last_token = None

    def build(self, **kwargs):
        """Builds the lexer from the specification. Must be
        called after the lexer object is created.

        This method exists separately, because the PLY
        manual warns against calling lex.lex inside __init__
        """
        self.lexer = lex.lex(object=self, **kwargs)

    def reset_lineno(self):
        """Resets the internal line number counter of the lexer."""
        self.lexer.lineno = 1

    def input(self, text):
        self.lexer.input(text)

    def token(self):
        self.last_token = self.lexer.token()
        return self.last_token

    def find_tok_column(self, token):
        """Find the column of the token in its line."""
        last_cr = self.lexer.lexdata.rfind("\n", 0, token.lexpos)
        return token.lexpos - last_cr

    # Internal auxiliary methods
    def _error(self, msg, token):
        location = self._make_tok_location(token)
        self.error_func(msg, location[0], location[1])
        self.lexer.skip(1)

    def _make_tok_location(self, token):
        return (token.lineno, self.find_tok_column(token))

    # Reserved keywords
    keywords = (
        "ASSERT",
        "BREAK",
        "CHAR",
        "ELSE",
        "FLOAT",
        "FOR",
        "IF",
        "INT",
        "PRINT",
        "READ",
        "RETURN",
        "VOID",
        "WHILE",
    )

    keyword_map = {}
    for keyword in keywords:
        keyword_map[keyword.lower()] = keyword

    #
    # All the tokens recognized by the lexer
    #
    tokens = keywords + (
        # Identifiers
        "ID",
        # constants
        "INT_CONST",
        "FLOAT_CONST",
        "CHAR_CONST",
        "STRING_LITERAL",
        # Operators and Delimiters:
        'NOT',
        'OR',
        'AND',
        'UNARY_AND',
        'EQ',
        'EQUALS',
        'DIVIDE',
        'TIMES',
        'NE',
        'LE',
        'GE',
        'LT',
        'GT',
        'PLUSPLUS',
        'DIVEQUAL',
        'PLUSEQUAL',
        'MINUSEQUAL',
        'TIMESEQUAL',
        'MODEQUAL',
        'MINUSMINUS',
        'PLUS',
        'MINUS',
        'MOD',
        'LPAREN',
        'RPAREN',
        'LBRACE',
        'RBRACE',
        'LBRACKET',
        'RBRACKET',
        'SEMI',
        'COMMA',
    )

    #
    # Rules
    #
    t_ignore = " \t"

    # Newlines
    def t_NEWLINE(self, t):
        r'(\n)'
        t.lexer.lineno += t.value.count("\n")

    def t_comment(self, t):
        r'(\/\*(((.)|(\n))*?)\*\/)|(\/\/(.*?)\n)'
        t.lexer.lineno += t.value.count("\n")

    def t_unterminated_comment_error(self, t):
        r'\/\*((.)|(\n))*'
        t.lexer.lineno += t.value.count("\n")
        self._error('Unterminated comment', t)

    def t_ID(self, t):
        r'([a-zA-Z]|(_))(([a-zA-Z0-9]|(_))*)'
        t.type = self.keyword_map.get(t.value, "ID")
        return t

    def t_OR(self, t):
        r'(\|\|)'
        t.type = 'OR'
        return t

    def t_AND(self, t):
        r'(&&)'
        t.type = 'AND'
        return t

    def t_LE(self, t):
        r'(<=)'
        t.type = 'LE'
        return t

    def t_GE(self, t):
        r'(>=)'
        t.type = 'GE'
        return t

    def t_DIVEQUAL(self, t):
        r'(\/=)'
        t.type = 'DIVEQUAL'
        return t

    def t_PLUSEQUAL(self, t):
        r'(\+=)'
        t.type = 'PLUSEQUAL'
        return t

    def t_MINUSEQUAL(self, t):
        r'(\-=)'
        t.type = 'MINUSEQUAL'
        return t

    def t_TIMESEQUAL(self, t):
        r'(\*=)'
        t.type = 'TIMESEQUAL'
        return t

    def t_MODEQUAL(self, t):
        r'(%=)'
        t.type = 'MODEQUAL'
        return t

    def t_PLUSPLUS(self, t):
        r'(\+\+)'
        t.type = 'PLUSPLUS'
        return t

    def t_MINUSMINUS(self, t):
        r'(--)'
        t.type = 'MINUSMINUS'
        return t

    def t_PLUS(self, t):
        r'(\+)'
        t.type = 'PLUS'
        return t

    def t_MINUS(self, t):
        r'(-)'
        t.type = 'MINUS'
        return t

    def t_EQ(self, t):
        r'(==)'
        t.type = 'EQ'
        return t

    def t_LT(self, t):
        r'(<)'
        t.type = 'LT'
        return t

    def t_GT(self, t):
        r'(>)'
        t.type = 'GT'
        return t

    def t_EQUALS(self, t):
        r'(=)'
        t.type = 'EQUALS'
        return t

    def t_UNARY_AND(self, t):
        r'(&)'
        t.type = 'UNARY_AND'
        return t

    def t_NE(self, t):
        r'(!=)'
        t.type = 'NE'
        return t

    def t_NOT(self, t):
        r'(!)'
        t.type = 'NOT'
        return t

    def t_DIVIDE(self, t):
        r'(\/)'
        t.type = 'DIVIDE'
        return t

    def t_TIMES(self, t):
        r"(\*)"
        t.type = 'TIMES'
        return t

    def t_MOD(self, t):
        r'(%)'
        t.type = 'MOD'
        return t

    def t_SEMI(self, t):
        r'(;)'
        t.type = "SEMI"
        return t

    def t_COMMA(self, t):
        r'(,)'
        t.type = 'COMMA'
        return t

    def t_LPAREN(self, t):
        r"(\()"
        t.type = 'LPAREN'
        return t

    def t_RPAREN(self, t):
        r"(\))"
        t.type = 'RPAREN'
        return t

    def t_LBRACE(self, t):
        r"({)"
        t.type = 'LBRACE'
        return t

    def t_RBRACE(self, t):
        r"(})"
        t.type = 'RBRACE'
        return t

    def t_LBRACKET(self, t):
        r"(\[)"
        t.type = "LBRACKET"
        return t

    def t_RBRACKET(self, t):
        r"(\])"
        t.type = "RBRACKET"
        return t

    def t_FLOAT_CONST(self, t):
        r"(([0-9]*)\.([0-9]+))|(([0-9]+)\.([0-9]*))"
        t.type = "FLOAT_CONST"
        return t

    def t_INT_CONST(self, t):
        r"([0-9]+)"
        t.type = "INT_CONST"
        return t

    def t_CHAR_CONST(self, t):
        r"('(.)')"
        t.type = "CHAR_CONST"
        return t

    def t_STRING_LITERAL(self, t):
        r'"(.*?)"'
        t.type = "STRING_LITERAL"
        t.value = re.findall(r'"(.*?)"', t.value)[0]
        return t

    def t_error(self, t):
        msg = "Illegal character %s" % repr(t.value[0])
        self._error(msg, t)

    # Scanner (used only for test)
    def scan(self, data):
        self.lexer.input(data)
        output = ""
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            print(tok)
            output += str(tok) + "\n"
        return output


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be scanned", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    def print_error(msg, x, y):
        # use stdout to match with the output in the .out test files
        print("Lexical error: %s at %d:%d" % (msg, x, y), file=sys.stdout)

    # set error function
    m = UCLexer(print_error)
    # Build the lexer
    m.build()
    # open file and print tokens
    with open(input_path) as f:
        m.scan(f.read())
