import sys
import tty
import termios


class GetKey:
    def __call__(self):
        # get device number
        fd = sys.stdin.fileno()
        # store the old settings to restore later
        old_settings = termios.tcgetattr(fd)
        try:
            # set buffer so that no 'press enter' is required
            tty.setraw(sys.stdin.fileno())
            # read the first input character
            ch = sys.stdin.read(1)
        finally:
            # restore former settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def get_manual_arrow_key():
    inkey = GetKey()
    print(inkey, flush=True)
    while(1):
        k = inkey()
        if k != '':
            break
    if k in '1234567':
        return int(k)
