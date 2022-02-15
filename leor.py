#!./venv/bin/python3

from audioop import reverse
import attr
import re
from enum import Enum
from typing import \
    Any, Literal, cast, Callable, Dict, List, Tuple, Union


#region Helpers
__auin_counter = 0
def auin(reset: bool=False) -> int:
    global __auin_counter
    if reset:
        __auin_counter = 0

    i = __auin_counter
    __auin_counter += 1
    return i
#endregion


#region Character Tests
ESCAPABLE = {
    "'": "'",
    "\"": "\"",
    "\\": "\\",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "0": "\0",
}


def is_not_newline(x: str) -> bool:
    return x != '\n'


def is_ws(x: str) -> bool:
    return re.match(r"\s", x) is not None


def is_comment(x: str) -> bool:
    return x == "#"


def is_quote(x: str) -> bool:
    return x in "\"\'"


def is_digit(x: str) -> bool:
    return re.match(r"\d", x) is not None


def is_id(x: str) -> bool:
    return re.match(r"[a-zA-Z0-9_?]", x) is not None


def is_op(x: str) -> bool:
    return re.match(r"[+\-*\/%=!^|&<>:]", x) is not None


def is_punc(x: str) -> bool:
    return re.match(r"[{[(;,.\\)\]}]", x) is not None
    


def is_kw(x: str) -> bool:
    return re.match(
        r"true|false|if|else|def|mut|const|return",
        x
    ) is not None
#endregion


#region CharStream
@attr.s(slots=True)
class CharStream:

    _it: int = attr.ib(init=False, default=0)
    _row: int = attr.ib(init=False, default=1)
    _col: int = attr.ib(init=False, default=1)
    _data: str = attr.ib(init=True)

    @property
    def peek(self) -> str:
        if self.eof():
            return "\0"
        return self._data[self._it]

    @property
    def next(self) -> str:
        if self.eof():
            return "\0"
        c = self.peek

        if c == "\n":
            self._row += 1
            self._col = 1
        else:
            self._col += 1

        self._it += 1
        return c

    @property
    def pos(self) -> Tuple[int, int]:
        return (self._row, self._col)

    def eof(self) -> bool:
        return self._it >= len(self._data)
#endregion


#region Token
class TokenType(Enum):
    NULL    = auin(True)
    EOF     = auin()
    INT     = auin()
    FLOAT   = auin()
    CHAR    = auin()
    STRING  = auin()
    VAR     = auin()
    KW      = auin()
    OP      = auin()
    PUNC    = auin()


@attr.s(slots=True, repr=False)
class Token:
    type: TokenType = attr.ib(init=True)
    value: str = attr.ib(init=True)
    pos: Tuple[int, int] = attr.ib(init=True)

    @classmethod
    def null_token(cls):
        return cls(TokenType.NULL, "", (0, 0))

    @classmethod
    def eof_token(cls, pos: Tuple[int, int]):
        return cls(TokenType.EOF, "", pos)

    def __repr__(self) -> str:
        return f"{self.type}, {self.value!r} @ {self.pos}"

    def __bool__(self) -> bool:
        return not self.type in [TokenType.NULL, TokenType.EOF]
#endregion


#region Lexer
@attr.s(slots=True)
class Lexer:

    data: CharStream = attr.ib(init=True)
    current: Token = attr.ib(
        init=False,
        default=Token.null_token()
    )

    def rd_while(self, pred: Callable[[str], bool]) -> str:
        ret = ""
        while not self.data.eof() and pred(self.data.peek):
            ret += self.data.next
        
        return ret

    def rd_esc(self, end: str) -> str:
        ret = ""
        esc = False
        while not self.data.eof():
            c = self.data.next
            if esc:
                if ESCAPABLE.get(c) is not None:
                    ret += ESCAPABLE[c]
                else:
                    ret += c
                esc = False
            elif c == end:
                break
            elif c == "\\":
                esc = True
            else:
                ret += c
        return ret

    def rd_number(self) -> Token:
        dot = False
        def f(c: str) -> bool:
            nonlocal dot

            if (c == "."):
                if dot:
                    return False
                dot = True
                return True
            return is_digit(c)

        pos = self.data.pos
        number = self.rd_while(f)
        return \
            Token(TokenType.FLOAT, number, pos) if dot \
            else Token(TokenType.INT, number, pos)

    def rd_id(self) -> Token:
        pos = self.data.pos
        x = self.rd_while(is_id)
        return \
            Token(TokenType.KW, x, pos) if is_kw(x) \
            else Token(TokenType.VAR, x, pos)

    def rd_string(self) -> Token:
        pos = self.data.pos
        quote = self.data.next
        escaped = self.rd_esc(quote)
        return \
            Token(TokenType.STRING, escaped, pos) if quote == "\"" \
            else Token(TokenType.CHAR, escaped, pos)

    def rd_next(self) -> Token:
        self.rd_while(is_ws)
        if self.data.eof():
            return Token(TokenType.EOF, "", self.data.pos)

        c = self.data.peek
        pos = self.data.pos

        if (is_comment(c)):
            self.rd_while(is_not_newline)
            return self.rd_next()
        
        if (is_digit(c)):
            return self.rd_number()
        if (is_id(c)):
            return self.rd_id()

        if (is_quote(c)):
            return self.rd_string()

        if (is_op(c)):
            return \
                Token(TokenType.OP, self.rd_while(is_op), pos)
        if (is_punc(c)):
            return \
                Token(TokenType.PUNC, self.data.next, pos)
        raise SyntaxError(f"ERROR: Unexpected character '{c}' @ {pos}")

    @property
    def peek(self) -> Token:
        if self.current.type == TokenType.NULL:
            self.current = self.rd_next()
        return self.current
    
    @property
    def next(self) -> Token:
        t = self.peek
        if t.type in [TokenType.NULL, TokenType.EOF]:
            return t
        self.current = Token.null_token()
        return t

    def eof(self) -> bool:
        return self.peek.type == TokenType.EOF
#endregion


#region AST 
class ASTType(Enum):
    NULL     = auin(True)
    BOOL     = auin()
    INT      = auin()
    FLOAT    = auin()
    CHAR     = auin()
    STRING   = auin()
    VAR      = auin()
    BINARY   = auin()
    ASSIGN   = auin()
    DECL     = auin()
    FUNCTION = auin()
    CALL     = auin()
    PROG     = auin()
    RETURN   = auin()


OP_PRECEDENCE: Dict[str, int] = {
    "=": 1,
    "||": 2,
    "&&": 3,
    "<": 7, ">": 7, "<=": 7, ">=": 7, "==": 7, "!=": 7,
    "+": 10, "-": 10,
    "*": 20, "/": 20, "%": 20,
}


OP_TYPE: Dict[str, ASTType] = {
    "=":   ASTType.ASSIGN,
    "||":  ASTType.BINARY,
    "&&":  ASTType.BINARY,
    "<":   ASTType.BINARY,
    ">":   ASTType.BINARY,
    "<=":  ASTType.BINARY,
    ">=":  ASTType.BINARY,
    "==":  ASTType.BINARY,
    "!=":  ASTType.BINARY,
    "+":   ASTType.BINARY,
    "-":   ASTType.BINARY,
    "*":   ASTType.BINARY,
    "/":   ASTType.BINARY,
    "%":   ASTType.BINARY,
}


@attr.s(slots=True)
class BasicAST:
    type: ASTType = attr.ib(init=True, default=ASTType.NULL)
    pos: Tuple[int, int] = attr.ib(init=True, default=(0, 0))


class Node:
    pass


@attr.s(slots=True)
class BoolNode(Node):
    value: bool = attr.ib(init=True)


@attr.s(slots=True)
class IntNode(Node):
    value: int = attr.ib(init=True)


@attr.s(slots=True)
class FloatNode(Node):
    value: float = attr.ib(init=True)


@attr.s(slots=True)
class CharNode(Node):
    value: str = attr.ib(init=True)


@attr.s(slots=True)
class StringNode(Node):
    value: str = attr.ib(init=True)


@attr.s(slots=True)
class VarNode(Node):
    value: str = attr.ib(init=True)


@attr.s(slots=True)
class BinaryNode(Node):
    left: BasicAST = attr.ib(init=True)
    right: BasicAST = attr.ib(init=True)
    op: str = attr.ib(init=True)


@attr.s(slots=True)
class DeclNode(Node):
    name: str = attr.ib(init=True)
    type: str = attr.ib(init=True)
    value: BasicAST | None = attr.ib(init=True)
    const: bool = attr.ib(init=True)


@attr.s(slots=True)
class FuncNode(Node):
    name: str = attr.ib(init=True)
    type: str = attr.ib(init=True)
    params: List[BasicAST] = attr.ib(init=True)
    body: BasicAST = attr.ib(init=True)


@attr.s(slots=True)
class CallNode(Node):
    func: BasicAST = attr.ib(init=True)
    args: List[BasicAST] = attr.ib(init=True)


@attr.s(slots=True)
class ProgNode(Node):
    prog: List[BasicAST] = attr.ib(init=True)


@attr.s(slots=True)
class ReturnNode(Node):
    value: BasicAST = attr.ib(init=True)


@attr.s(slots=True)
class AST(BasicAST):
    node: Union[
        Node, None
    ] = attr.ib(init=True, default=None)
#endregion


#region Parser
@attr.s(slots=True)
class Parser:
    data: Lexer = attr.ib(init=True)

    def __call__(self) -> AST:
        return self.parse_toplevel()

    def is_punc(self, x: Union[str, None] = None) -> Token:
        tok = self.data.peek
        if (
            tok.type == TokenType.PUNC
            and (x is None or tok.value == x)
        ):
            return tok
        return Token.null_token()

    def is_op(self, x: Union[str, None] = None) -> Token:
        tok = self.data.peek
        if (
            tok.type == TokenType.OP
            and (x is None or tok.value == x)
        ):
            return tok
        return Token.null_token()

    def is_kw(self, x: Union[str, None] = None) -> Token:
        tok = self.data.peek
        if (
            tok.type == TokenType.KW
            and (x is None or tok.value == x)
        ):
            return tok
        return Token.null_token()

    def skip_punc(self, x: str) -> None:
        if self.is_punc(x):
            self.data.next
        else:
            raise SyntaxError(f"ERROR: Expected punctuation '{x}', but got {self.data.peek!r}")

    def skip_op(self, x: str) -> None:
        if self.is_op(x):
            self.data.next
        else:
            raise SyntaxError(f"ERROR: Expected operator '{x}', but got {self.data.peek!r}")

    def skip_kw(self, x: str) -> None:
        if self.is_kw(x):
            self.data.next
        else:
            raise SyntaxError(f"ERROR: Expected keyword '{x}', but got {self.data.peek!r}")

    def delimited_str(
        self,
        beg: str,
        end: str,
        sep: str,
        parser: Callable[[], str],
    ) -> List[str]:
        ret: List[str] = []
        first = True

        self.skip_punc(beg)
        while not self.data.eof():
            if self.is_punc(end):
                break
            if first:
                first = False
            else:
                self.skip_punc(sep) # if self.is_punc(sep) else None
            if self.is_punc(end):
                break
            ret.append(parser())

        self.skip_punc(end)
        return ret

    def delimited_ast(
        self,
        beg: str,
        end: str,
        sep: str,
        parser: Callable[[], AST],
    ) -> List[BasicAST]:
        ret: List[BasicAST] = []
        first = True

        self.skip_punc(beg)
        while not self.data.eof():
            if self.is_punc(end):
                break
            if first:
                first = False
            else:
                self.skip_punc(sep) # if self.is_punc(sep) else None
            if self.is_punc(end):
                break
            ret.append(parser())

        self.skip_punc(end)
        return ret

    def parse_atom(self) -> AST:
        def f() -> AST:
            nonlocal self
            if self.is_punc("("):
                self.data.next
                exp: AST = self.parse_expression()
                self.skip_punc(")")
                return exp
            if self.is_punc("{"):
                return self.parse_prog()
            if self.is_kw("true") or self.is_kw("false"):
                return self.parse_bool()
            if self.is_kw("const") or self.is_kw("mut"):
                return self.parse_decl()
            if self.is_kw("def"):
                return self.parse_def()
            if self.is_kw("return"):
                pos = self.data.next.pos
                return AST(
                    ASTType.RETURN, pos, ReturnNode(self.parse_expression())
                )
            tok = self.data.next
            match tok.type:
                case TokenType.VAR:
                    return AST(ASTType.VAR, tok.pos, VarNode(tok.value))
                case TokenType.STRING:
                    return AST(ASTType.STRING, tok.pos, StringNode(tok.value))
                case TokenType.CHAR:
                    return AST(ASTType.CHAR, tok.pos, CharNode(tok.value))
                case TokenType.INT:
                    return AST(ASTType.INT, tok.pos, IntNode(int(tok.value)))
                case TokenType.FLOAT:
                    return AST(ASTType.FLOAT, tok.pos, FloatNode(float(tok.value)))
                case _:
                    raise SyntaxError(f"ERROR: Unexpected token {tok!r}")
            # MyPy complains about missing return statement
            # So here it is, it will never be hit
            # But it does not complain anymore
            return Token.null_token()

        return self.maybe_call(f)

    def parse_toplevel(self) -> AST:
        pos = self.data.peek.pos
        prog: List[BasicAST] = []
        while not self.data.eof():
            prog.append(self.parse_expression())
            self.skip_punc(";")
        return AST(ASTType.PROG, pos, ProgNode(prog))

    def parse_bool(self) -> AST:
        tok = self.data.next
        return AST(ASTType.BOOL, tok.pos, BoolNode(tok.value == "true"))

    def parse_decl(self) -> AST:
        pos = self.data.peek.pos
        const = False
        if self.is_kw("const"):
            const = True
        self.data.next
        name: str = self.parse_varname()
        self.skip_op(":")
        tp: str = self.parse_varname()
        value: AST | None = None
        if self.is_op("="):
            self.data.next
            value = self.parse_expression()

        return AST(
            ASTType.DECL, pos, DeclNode(name, tp, value, const)
        )
    
    def parse_def(self) -> AST:
        pos = self.data.next.pos
        name: str = self.parse_varname()
        params = self.delimited_ast(
            "(", ")", ",",
            self.parse_decl
        )
        self.skip_op("->")
        # TODO: Consider parsing types into AST instead of str
        tp: str = self.parse_varname()
        prog: AST = self.parse_prog()
        return AST(
            ASTType.FUNCTION,
            pos,
            FuncNode(
                name,
                tp,
                params,
                prog
            )
        )

    def parse_prog(self) -> AST:
        pos = self.data.peek.pos
        prog = self.delimited_ast("{", "}", ";", self.parse_expression)
        return AST(ASTType.PROG, pos, ProgNode(prog))

    def parse_expression(self) -> AST:
        def f() -> AST:
            nonlocal self
            return self.maybe_binary(self.parse_atom(), 0)
        
        return self.maybe_call(f)

    def parse_varname(self) -> str:
        name = self.data.next
        if name.type != TokenType.VAR:
            raise SyntaxError(f"ERROR: Expected varname but got {name!r}")
        return name.value

    def parse_call(self, function: AST) -> AST:
        pos = self.data.peek.pos
        return AST(
            ASTType.CALL,
            pos,
            CallNode(
                function,
                self.delimited_ast(
                    "(", ")", ",",
                    self.parse_expression
                )
            )
        )

    def maybe_call(self, expr: Callable[[], AST]) -> AST:
        e = expr()
        return self.parse_call(e) if self.is_punc("(") else e

    def maybe_binary(self, left: AST, my_prec: int) -> AST:
        pos = self.data.peek.pos
        tok = self.is_op()
        if tok:
            his_prec = OP_PRECEDENCE[tok.value]
            if his_prec > my_prec:
                self.data.next
                right = self.maybe_binary(self.parse_atom(), his_prec)
                binary = AST(
                    OP_TYPE[tok.value],
                    pos,
                    BinaryNode(
                        left, right, tok.value
                    )
                )
                return self.maybe_binary(binary, my_prec)
        return left
#endregion


#region Program
class Addr(int):
    pass


class Size(int):
    pass


@attr.s(slots=True)
class Memory:

    capacity: int = attr.ib(default=0)
    allocated: Dict[Addr, Size] = attr.ib(default={})
    unallocated: List[Tuple[Addr, Size]] = attr.ib(default=[])

    def find_free_unallocated(self, size: Size) -> Addr | None:
        for i, v in enumerate(self.unallocated):
            (a, s) = v
            if size <= s:
                if size < s:
                    self.unallocated.append((Addr(a+size), Size(s-size)))
                self.unallocated.pop(i)
                i -= 1
                self.allocated[a] = size
                return a
        return None
                

    def alloc(self, size: Size) -> Addr:
        a = self.find_free_unallocated(size)
        if a is not None:
            return a
        a = Addr(self.capacity)
        self.allocated[a] = size
        self.capacity += size
        return a

    def free(self, addr: Addr):
        try:
            size = self.allocated.pop(addr)
        except:
            assert False, "TODO: Handle not in memory"
        
        self.unallocated.append((addr, size))


class Register(Enum):

    AX  = auin(True)
    BX  = auin()
    CX  = auin()
    DX  = auin()
    DI  = auin()
    SI  = auin()
    R8  = auin()
    R9  = auin()
    R10 = auin()
    R11 = auin()
    R12 = auin()
    R13 = auin()
    R14 = auin()
    R15 = auin()
    SP  = auin()
    BP  = auin()

    STACK = auin()


class Regs:
    reg_qword: Dict[Register, str] = {
        Register.AX:  "rax",
        Register.BX:  "rbx",
        Register.CX:  "rcx",
        Register.DX:  "rdx",
        Register.DI:  "rdi",
        Register.SI:  "rsi",
        Register.R8:  "r8",
        Register.R9:  "r9",
        Register.R10: "r10",
        Register.R11: "r11",
        Register.R12: "r12",
        Register.R13: "r13",
        Register.R14: "r14",
        Register.R15: "r15",
        Register.SP:  "rsp",
        Register.BP:  "rbp"
    }
    reg_dword: Dict[Register, str] = {
        Register.AX:  "eax",
        Register.BX:  "ebx",
        Register.CX:  "ecx",
        Register.DX:  "edx",
        Register.DI:  "edi",
        Register.SI:  "esi",
        Register.R8:  "r8d",
        Register.R9:  "r9d",
        Register.R10: "r10d",
        Register.R11: "r11d",
        Register.R12: "r12d",
        Register.R13: "r13d",
        Register.R14: "r14d",
        Register.R15: "r15d",
        Register.SP:  "esp",
        Register.BP:  "ebp"
    }
    reg_word: Dict[Register, str] = {
        Register.AX:  "ax",
        Register.BX:  "bx",
        Register.CX:  "cx",
        Register.DX:  "dx",
        Register.DI:  "di",
        Register.SI:  "si",
        Register.R8:  "r8w",
        Register.R9:  "r9w",
        Register.R10: "r10w",
        Register.R11: "r11w",
        Register.R12: "r12w",
        Register.R13: "r13w",
        Register.R14: "r14w",
        Register.R15: "r15w",
        Register.SP:  "sp",
        Register.BP:  "bp"
    }
    reg_byte: Dict[Register, str] = {
        Register.AX:  "al",
        Register.BX:  "bl",
        Register.CX:  "cl",
        Register.DX:  "dl",
        Register.DI:  "dil",
        Register.SI:  "sil",
        Register.R8:  "r8b",
        Register.R9:  "r9b",
        Register.R10: "r10b",
        Register.R11: "r11b",
        Register.R12: "r12b",
        Register.R13: "r13b",
        Register.R14: "r14b",
        Register.R15: "r15b",
        Register.SP:  "spl",
        Register.BP:  "bpl"
    }

    reg_args: List[Register] = [
        Register.DI,
        Register.SI,
        Register.DX,
        Register.CX,
        Register.R8,
        Register.R9,
        Register.STACK
    ]

    @classmethod
    def q(cls, reg: Register) -> str:
        return cls.reg_qword[reg]

    @classmethod
    def d(cls, reg: Register) -> str:
        return cls.reg_dword[reg]

    @classmethod
    def w(cls, reg: Register) -> str:
        return cls.reg_word[reg]

    @classmethod
    def b(cls, reg: Register) -> str:
        return cls.reg_byte[reg]

    @classmethod
    def fs(cls, size: Literal[1, 2, 4, 8], reg: Register) -> str:
        match size:
            case 1:
                return cls.b(reg)
            case 2:
                return cls.w(reg)
            case 4:
                return cls.d(reg)
            case 8:
                return cls.q(reg)

    @classmethod
    def arg(cls, idx: int) -> Register:
        if idx >= len(cls.reg_args):
            return Register.STACK
        return cls.reg_args[idx]


class BasicType:
    pass


@attr.s(slots=True, frozen=True)
class Type(BasicType):
    name: str = attr.ib(init=True)
    size: int = attr.ib(init=True)

    binary_ops: dict[str, dict[BasicType, Callable[[Any, Register], str]]] = attr.ib(default={})

    def op(
        self, op: str,
        type_right: BasicType,
        pos: tuple[int, int]
    ) -> Callable[[Any, Register], str]:
        row, col = pos
        right = cast(type(self), type_right)

        ops = self.binary_ops.get(op)

        if ops is None:
            raise CompilationError(
                f"{row}:{col}:ERR: Operator '{op}' is not overloaded for type {self.name}"
            )
        
        op_asm = ops.get(right)
        if op_asm is None:
            raise CompilationError(
                f"{row}:{col}:ERR: Operator '{op}' is not overloaded for types {self.name} and {right.name}"
            )
        
        return op_asm    


@attr.s(slots=True)
class Var:
    index: Addr = attr.ib(init=True)
    size: Size = attr.ib(init=True)
    # TODO: Type structure
    type: Type = attr.ib(init=True)
    const: bool = attr.ib(init=True)


@attr.s(slots=True)
class Env:
    name: str = attr.ib(init=True)
    children: List[str] = attr.ib(init=True)
    parent: str | None = attr.ib(init=True)
    variables: Dict[str, Var] = attr.ib(init=True)

    # TODO: add proper types
    params: Dict[str, str] = attr.ib(init=False, default={})


@attr.s
class Program:
    # Segments
    code: str = attr.ib(default="")
    data: str = attr.ib(default="")
    bss:  str = attr.ib(default="")

    envs: Dict[str, Env] = attr.ib(
        default={ "": Env("", [], None, {}) }
    ) # Global scope
    current_env_name: str = attr.ib(default="") # Global scope

    mem: Memory = attr.ib(default=Memory())

    data_index: int = attr.ib(default=0)

    @property
    def current_env(self) -> Env | None:
        return self.envs.get(self.current_env_name)

    def lookup_var(self, var_name: str, env_name: str | None = None) -> Var | None:
        env: Env | None = self.envs.get(self.current_env_name) if env_name is None else self.envs.get(env_name)
        assert env is not None, "Compiler error: not in any scope"

        while env is not None:
            var = env.variables.get(var_name)
            if var is not None:
                return var
            env = self.envs.get(env.parent) if env.parent is not None else None
        
        return None


    def lookup_func(self, func_name: str, env_name: str | None = None) -> Env | None:
        env: Env | None = self.envs.get(self.current_env_name) if env_name is None else self.envs.get(env_name)
        assert env is not None, "Compiler error: not in any scope"

        while env is not None:
            if func_name == env.name:
                return env
            if func_name in env.children:
                return self.envs[func_name]
            env = self.envs.get(env.parent) if env.parent is not None else None

        return None
#endregion


#region Compiler
class CompilationError(Exception):
    pass


def fasm_linux_x86_64_compile_node(prog: Program, ast: AST, reg: Register):
    assert prog.current_env, "Program has no environments"
    assert ast.node, "Cannot compile ast of none"

    row, col = ast.pos

    to_pop: list[Register] = []

    match ast.type:
        case ASTType.INT:
            intnode: IntNode = cast(IntNode, ast.node)

            prog.code += f"  mov {Regs.q(reg)}, {intnode.value}\n"
        case ASTType.STRING:
            strnode: StringNode = cast(StringNode, ast.node)
            string = ", ".join([hex(ord(c)) for c in strnode.value] + ["0x0"])

            prog.data += f"STR_{prog.data_index} db {string}\n"
            prog.code += f"  mov {Regs.q(reg)}, STR_{prog.data_index}\n"

            prog.data_index += 1
            
        case ASTType.DECL:
            declnode: DeclNode = cast(DeclNode, ast.node)
            
            if prog.lookup_func(declnode.name) or prog.lookup_var(declnode.name):
                raise CompilationError(f"ERROR: Variable {declnode.name} already exists")

            idx = prog.mem.alloc(Size(8))

            var = Var(idx, Size(8), declnode.type, declnode.const)
            prog.current_env.variables[declnode.name] = var 

            if declnode.const and not declnode.value:
                raise CompilationError(f"ERROR: Constant variable has to be initialized; '{declnode.name}' {ast.pos}")

            if declnode.value:
                fasm_linux_x86_64_compile_node(prog, cast(AST, declnode.value), reg)
                prog.code += f"  mov qword[mem+{idx}], {Regs.q(reg)}\n"

        case ASTType.PROG:
            prognode: ProgNode = cast(ProgNode, ast.node)
            for x in prognode.prog:
                fasm_linux_x86_64_compile_node(prog, cast(AST, x), reg)
        case ASTType.FUNCTION:
            funcnode: FuncNode = cast(FuncNode, ast.node)
            if prog.lookup_func(funcnode.name) or prog.lookup_var(funcnode.name):
                raise CompilationError(f"ERROR: Variable {funcnode.name} already exists")

            # TODO: types
            prog.envs[funcnode.name] = Env(
                funcnode.name, [], prog.current_env_name, {}
            )
            prog.current_env.children.append(funcnode.name)
            prog.current_env_name = funcnode.name

            for x in funcnode.params:
                fasm_linux_x86_64_compile_node(prog, cast(AST, x), reg)

            # Add every existing variables to params.
            # There are no variables other than params,
            # because we don't compile body yet
            for name, var in prog.current_env.variables.items():
                prog.current_env.params[name] = var.type

            prog.code += f"{funcnode.name}:\n"
            fasm_linux_x86_64_compile_node(prog, cast(AST, funcnode.body), reg)
            prog.code += f"  ret\n"

            prog.current_env_name = (
                prog.current_env.parent
                if prog.current_env.parent
                else ""
            )

        case ASTType.BINARY:
            binnode: BinaryNode = cast(BinaryNode, ast.node)

            if reg != Register.DI:
                prog.code += f"  push {Regs.q(Register.DI)}\n"
                to_pop.append(Register.DI)
            if reg != Register.SI:
                prog.code += f"  push {Regs.q(Register.SI)}\n"
                to_pop.append(Register.SI)

            if (
                binnode.left.type      in [ASTType.BINARY, ASTType.ASSIGN]
                or
                binnode.right.type not in [ASTType.BINARY, ASTType.ASSIGN]
            ):
                fasm_linux_x86_64_compile_node(
                    prog, cast(AST, binnode.left), Register.AX
                )
                prog.code += f"  mov {Regs.q(Register.DI)}, {Regs.q(Register.AX)}\n"
                fasm_linux_x86_64_compile_node(
                    prog, cast(AST, binnode.right), Register.AX
                )
                prog.code += f"  mov {Regs.q(Register.SI)}, {Regs.q(Register.AX)}\n"

            else:
                fasm_linux_x86_64_compile_node(
                    prog, cast(AST, binnode.right), Register.AX
                )
                prog.code += f"  mov {Regs.q(Register.SI)}, {Regs.q(Register.AX)}\n"

                fasm_linux_x86_64_compile_node(
                    prog, cast(AST, binnode.left), Register.AX
                )
                prog.code += f"  mov {Regs.q(Register.DI)}, {Regs.q(Register.AX)}\n"

            match binnode.op:
                case "+":
                    prog.code += f"  add {Regs.q(Register.DI)}, {Regs.q(Register.SI)}\n"
                case _:
                    assert False, \
                        f"Compilation of operator '{binnode.op}' is not implemented"
            prog.code += f"  mov {Regs.q(reg)}, {Regs.q(Register.DI)}\n"

            prog.code += f"  pop {Regs.q(Register.DI)}\n  pop {Regs.q(Register.SI)}\n"

        case ASTType.CALL:
            callnode: CallNode = cast(CallNode, ast.node)
            varnode: VarNode = cast(
                VarNode, cast(AST, callnode.func).node
            )

            # TODO: make variables callable
            func = prog.lookup_func(varnode.value)
            if not func:
                raise CompilationError(
                    f"{row}:{col}:ERR: Function '{varnode.value}' does not exist\n"
                )
            
            # TODO: typecheck parameters
            if len(callnode.args) != len(func.params):
                raise CompilationError(
                    f"{row}:{col}:ERR: Function '{varnode.value}' expected \
                      {len(func.params)} but got {len(callnode.args)}"
                )

            for i, arg in enumerate(callnode.args):
                reg_arg = Regs.arg(i)
                prog.code += f"  push {Regs.q(reg_arg)}\n"
                fasm_linux_x86_64_compile_node(
                    prog, cast(AST, arg), reg_arg
                )
                to_pop.append(reg_arg)
            
            prog.code += f"  call {varnode.value}\n"
            prog.code += f"  mov {Regs.q(reg)}, rax\n"

        case ASTType.RETURN:
            retnode: ReturnNode = cast(ReturnNode, ast.node)
            fasm_linux_x86_64_compile_node(
                prog, cast(AST, retnode.value), Register.AX)

        case _:
            assert False, f"Compilation of {ast.type} is not implemented yet\n"

    for pop in to_pop[::-1]:
        prog.code += f"  pop {Regs.q(pop)}\n"


def compile_fasm_linux_x86_64(ast: AST) -> Program:

    if ast.type != ASTType.PROG:
        raise CompilationError(
            f"Given AST should be a toplevel one"
        )

    prog = Program()

    prog.code += """\
format ELF64 executable 3
entry __start

segment readable executable
write:
  mov rax, 1
  syscall
  ret
__start:
  call main
  mov rdi, rax
  mov rax, 60
  syscall
"""

    # add built-in functions

    #write
    env_write = Env("write", [], prog.current_env_name, {})
    env_write.params = {"fd": "Int", "buf": "Char", "Count": "Int"}
    prog.current_env.children.append("write") if prog.current_env else 1
    prog.envs["write"] = env_write

    prog.data = "segment readable\n"
    prog.bss  = "segment readable writeable\n"

    fasm_linux_x86_64_compile_node(prog, ast, Register.AX)

    prog.bss += f"mem rb {prog.mem.capacity}\n"

    return prog
#endregion


def main():
    from argparse import ArgumentParser
    from os.path import abspath, basename, isfile 
    from subprocess import Popen

    argp = ArgumentParser(
        description="LEOR compiler"
    )

    argp.add_argument(
        "file", metavar="FILENAME", type=str,
        help="Source file name to compile"
    )

    argp.add_argument(
        "-r", action="store_true",
        help="Run binary after compiling"
    )

    args = argp.parse_args()

    path: str = abspath(args.file)
    name: str = basename(path)

    name = name.split(".")[0]

    if not isfile(path):
        raise FileNotFoundError(f"ERROR: File '{path}' does not exist")
    
    print("[INFO] Compiling source...")

    cs: CharStream

    with open(path, "r") as f:
        cs = CharStream(f.read())

    lx = Lexer(cs)
    parser = Parser(lx)

    prog = compile_fasm_linux_x86_64(parser())

    print("[INFO] Generating assembly...")

    with open(f"{name}.asm", "w") as f:
        f.write(f"{prog.code}{prog.data}{prog.bss}")

    print("[CMD] Running fasm...")
    process = Popen(["fasm", f"{name}.asm"])
    process.wait()

    if (args.r):
        print("[CMD] Running binary...\n")
        process = Popen([f"./{name}"])
        process.wait()


if __name__=="__main__":
    main()
