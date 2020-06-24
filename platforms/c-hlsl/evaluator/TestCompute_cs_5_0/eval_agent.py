#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import logging
import subprocess

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import tornado.ioloop
import tornado.web
from tornado import options

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
        
        cmd = ["./TestCompute.exe"]
        logging.info("Starting execute: '%s'", ' '.join(cmd))
        output = subprocess.check_output(cmd, timeout=5)
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
    
    app = tornado.web.Application([(r"/post", POSTHandler), (r"/(.*)", PUTHandler)])
    app.listen(port)
    logging.info("Service is listening on ':%d'", port)
    
    tornado.ioloop.IOLoop.current().start()
