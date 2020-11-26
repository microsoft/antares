#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import logging
import subprocess
import urllib.request

# Assum this script is running in WSL 1.0
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    cs_compiler_path = 'C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319\\csc.exe'
else:
    cs_compiler_path = '/mnt/c/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe'


import tornado.ioloop
import tornado.web
from tornado import options

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
cmd = ["sh", "-c", "./TestCompute.exe || true"]

@tornado.web.stream_request_body
class PUTHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.bytes_read = 0
        
    def prepare(self):
        self.content = []

    def data_received(self, chunk):
        self.bytes_read += len(chunk)
        self.content.append(chunk)

    def put(self, filename):
        mtype = self.request.headers.get("Content-Type")
        logging.info('PUT "%s" "%s" %d bytes', filename, mtype, self.bytes_read)

        file = "dx_kernel.hlsl"
        self.fp = open(file, "wb")
        for chunk in self.content:
            self.fp.write(chunk)
        self.fp.close()
        
        logging.info("Starting execute: '%s'", str(cmd))
        output = subprocess.check_output(cmd, timeout=20)
        self.write(output)
        

class POSTHandler(tornado.web.RequestHandler):
    def post(self):
        for field_name, files in self.request.files.items():
            for info in files:
                filename, content_type = info["filename"], info["content_type"]
                body = info["body"]
                logging.info(
                    'POST "%s" "%s" %d bytes', filename, content_type, len(body)
                )
        self.write("Not implement.")

if __name__ == "__main__":
    port = 6000
    if not os.path.exists('./antares_hlsl_v0.1_x64.dll'):
      logging.info("Service is downloading `antares_hlsl_v0.1_x64.dll` ..")
      dll_data = urllib.request.urlopen("https://github.com/microsoft/antares/raw/library/antares_hlsl_v0.1_x64.dll").read()
      with open('antares_hlsl_v0.1_x64.dll', 'wb') as fp:
        fp.write(dll_data)
      logging.info("`antares_hlsl_v0.1_x64.dll` is created successfully.")

    logging.info("Service is compiling evaluator `TestCompute.cs` ..")
    assert 0 == os.system("%s /out:TestCompute.exe TestCompute.cs" % cs_compiler_path), "Failed to compile `TestCompute.cs` for HLSL shader evaluation."
    logging.info("HLSL evaluator is created successfully.")
    assert 0 == os.system("/bin/chmod a+x TestCompute.exe")
    assert 0 == os.system("rm -f dx_kernel.hlsl")
    subprocess.check_output(cmd, timeout=20)
    
    app = tornado.web.Application([(r"/post", POSTHandler), (r"/(.*)", PUTHandler)])
    app.listen(port)
    logging.info("Service is listening on ':%d'", port)
    
    tornado.ioloop.IOLoop.current().start()
