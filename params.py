#!/usr/bin/env python3
import clang.cindex as ci

support = [
  ("NV0000_CTRL_CMD_GPU_ATTACH_IDS", "NV0000_CTRL_GPU_ATTACH_IDS_PARAMS"),
]

args = [
  "-Iopen-gpu-kernel-modules/src/common/sdk/nvidia/inc",
]

lookup = {}
def walk(node):
  if node.kind == ci.CursorKind.TYPEDEF_DECL: lookup[node.spelling] = node
  for n in node.get_children(): walk(n)

fmt_strs = {
  "NvHandle": "",
}

def write_field(f, field):
  ll = 30
  match field.kind:
    case ci.CursorKind.UNION_DECL: f.write(f'  printf("%{ll}s: <union>\\n", "{field.spelling}");\n')
    case ci.CursorKind.FIELD_DECL:
      typeref, is_array = None, False
      for n in field.get_children():
        if n.kind == ci.CursorKind.TYPE_REF: typeref = n.spelling
        if n.kind == ci.CursorKind.INTEGER_LITERAL: is_array = True
      if is_array: f.write(f'  printf("%{ll}s: <{typeref} is array> ", "{field.spelling}");\n')
      elif typeref in fmt_strs: f.write(f'  printf("%{ll}s: {fmt_strs[typeref]}\\n", "{field.spelling}", p->{field.spelling});\n')
      else: f.write(f'  printf("%{ll}s: <{typeref} not parsed>\\n", "{field.spelling}");\n')


if __name__ == "__main__":
  index = ci.Index.create()
  tu = index.parse("stub.c", args=args)
  walk(tu.cursor)
  with open("params.h", "w") as f:
    f.write("#pragma once\n\n")
    for n, t in support:
      f.write(f"static void params_{n}(void *_p) {{\n")
      f.write(f"  {t} *p = ({t} *)_p;\n")
      f.write(f"  printf(\"{n}\\n\");\n")
      for x in list(lookup[t].get_children())[0].get_children():
        write_field(f, x)
      f.write("}\n\n")

