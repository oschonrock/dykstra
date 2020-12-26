# The MIT License (MIT)
#
# Copyright (c) 2018 Mateusz Pusz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from conans import ConanFile, CMake, tools
from conans.tools import load, Version
from conans.errors import ConanInvalidConfiguration
import re
import os


def get_version():
    try:
        content = load("src/CMakeLists.txt")
        version = re.search(r"project\([^\)]+VERSION (\d+\.\d+\.\d+)[^\)]*\)", content).group(1)
        return version.strip()
    except Exception:
        return None


class StampConan(ConanFile):
    name = "CHANGEME"
    version = get_version()
    author = "CHANGEME"
    license = "GPL"
    url = "none"
    description = "Stamp is a boilerplate temlate for c++ projects - CHANGEME"
    exports = ["LICENSE.md"]
    exports_sources = ["src/*", "test/*", "cmake/*", "example/*","CMakeLists.txt"]
    settings = "os", "compiler", "build_type", "arch"
    requires = (
        # "range-v3/0.10.0@ericniebler/stable",
    )
    generators = "cmake"

    @property
    def _run_tests(self):
        return tools.get_env("CONAN_RUN_TESTS", False)

    def package(self):
        self.copy(pattern="LICENSE.md", dst="licenses")
        cmake = self._configure_cmake()
        cmake.install()

    def package_id(self):
        self.info.settings.clear()
        self.info.settings.compiler = self.settings.compiler
        self.info.settings.compiler.version = self.settings.compiler.version
