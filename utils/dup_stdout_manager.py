import sys


class DupStdoutFileWriter(object):
    def __init__(self, stdout, path, mode, summary_writer):
        self.path = path
        self._content = ''
        self._stdout = stdout
        self._file = open(path, mode)
        self._summary_writer = summary_writer

    def write(self, msg):
        while '\n' in msg:
            pos = msg.find('\n')
            self._content += msg[:pos + 1]
            self.flush()
            msg = msg[pos + 1:]
        self._content += msg
        if len(self._content) > 1000:
            self.flush()

    def flush(self):
        self._stdout.write(self._content)
        self._stdout.flush()
        self._file.write(self._content)
        self._file.flush()
        if self._summary_writer is not None:
            self._summary_writer.add_text('cmd', self._content)
        self._content = ''

    def __del__(self):
        self._file.close()


class DupStdoutFileManager(object):
    def __init__(self, path, mode='w+', summary_writer=None):
        self.path = path
        self.mode = mode
        self.summary_writer = summary_writer

    def __enter__(self):
        self._stdout = sys.stdout
        self._file = DupStdoutFileWriter(self._stdout, self.path, self.mode, self.summary_writer)
        sys.stdout = self._file

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout