from enum import Enum

LLAMA2_INST_CLOSE = "[/INST]\n"

# Standard Llama2 template
TEMP_LLAMA2 = """
[INST]<<SYS>>
{sys}

{instr}

{input}
[/INST]
"""


SYS_1 = """You are a technical documentation writer. You always write clear, concise, and accurate documentation for
scientific experiments. Your documentation focuses on the experiment's purpose, procedure, and results. Therefore,
details about specific python functions, packages, or libraries are not necessary. Your readers are experimental
scientists.
"""

INSTR_SWEETP_1 = """Please generate high-level two paragraph documentation for the following experiment. The first
paragraph should explain the purpose and the second one the procedure, but don't use the word 'Paragraph'"""


class SystemPrompts(Enum):
    SYS_1 = "SYS_1"


class InstructionPrompts(Enum):
    INSTR_SWEETP_1 = "INSTR_SWEETP_1"


SYS = {SystemPrompts.SYS_1: SYS_1}
INSTR = {InstructionPrompts.INSTR_SWEETP_1: INSTR_SWEETP_1}
