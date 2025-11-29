print(ord('牛'))
print(chr(29275))
print(chr(0))
print(repr(chr(0)))

# Remark:
# repr 是程序员看的“代码形式/转义形式”，print 是用户看到的“实际显示效果”。
# repr 会把 chr(0) 显示为可见的转义符 \x00，而 print 显示时完全不可见。