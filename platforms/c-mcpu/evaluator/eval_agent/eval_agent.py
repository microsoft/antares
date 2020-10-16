#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import logging
import subprocess
import platform

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import tornado.ioloop
import tornado.web
from tornado import options

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def generate_source_file(kernel_code):
    rank = 0
    pos = kernel_code.find('///')
    if pos == -1:
        return False, rank
    pos += len('///')

    sep = kernel_code.find(':', pos)
    if sep == -1:
        return False, rank

    tail = kernel_code.find('\n', sep)
    if tail == -1:
        return False, rank

    input_str = kernel_code[pos:sep]
    output_str = kernel_code[sep + 1:tail]

    inputs = list(filter(None, input_str.split(',')))
    outputs = list(filter(None, output_str.split(',')))

    rank_pattern = '__rank__ = '
    pos = kernel_code.find(rank_pattern)
    if pos == -1:
        return False, rank

    tail = kernel_code.find('\n', pos)
    if tail == -1:
        return False, rank
    rank = int(kernel_code[pos + len(rank_pattern):tail])
    
    # Insert rank parameter.
    pos = kernel_code.find('kernel_main')
    if pos == -1:
        return False, rank
    tail = kernel_code.find(')', pos)
    if tail == -1:
        return False, rank
    kernel_code = kernel_code[:tail] + ', int __rank__' + kernel_code[tail:]

    # Input and output.
    type_converter = {
        "int32": "int",
        "float32": "float",
        "byte": "unsigned char"
    }
    main_func_body = ""
    input_types = [];
    output_types = [];
    for i, input in enumerate(inputs):
        shape, type, name = input.split('/')
        dims = list(filter(None, shape.split('-')))
        size = 1
        for dim in dims:
            size *= int(dim)
        idx = type.find("@")
        if idx >= 0:
            input_types.append(type[:idx])
            bits = int(type[idx + 1:])
            type = "byte"
            size = size * bits / 8
        else:
            input_types.append(type_converter[type])
        if not type in type_converter:
            return False, rank
        main_func_body += (type_converter[type] + '* input' + str(i) + ' = new ' + type_converter[type] + '[' + str(int(size)) + '];\n')
        main_func_body += ('for( int i = 0; i < ' + str(int(size)) + '; ++i) {\n')
        if idx >= 0:
            main_func_body += ('    input{}[i] = 0;\n'.format(i))
        else:
            main_func_body += ('    input{}[i] = ({} + 1 + i) % 71;\n'.format(i, i))
        main_func_body += ('}\n')
    for i, output in enumerate(outputs):
        shape, type, name = output.split('/')
        dims = list(filter(None, shape.split('-')))
        size = 1
        for dim in dims:
            size *= int(dim)
        idx = type.find("@")
        if idx >= 0:
            output_types.append(type[:idx])
            bits = int(type[idx + 1:])
            type = "byte"
            size = size * bits / 8
        else:
            output_types.append(type_converter[type])
        if not type in type_converter:
            return False, rank
        main_func_body += (type_converter[type]+ '* output' + str(i) + ' = new ' + type_converter[type]+ '[' + str(int(size)) + '];\n\n')

    # Threadpool.
    main_func_body += ('ThreadPool pool(' + str(rank) + ');\n')
    main_func_body += ('std::vector< std::future<void> > results;\n')
    main_func_body += ('results.reserve(' + str(rank) + ');\n\n')

    # Lambda for testing kernel and timing.
    main_func_body += ('std::chrono::high_resolution_clock::time_point t1, t2;\n')
    main_func_body += ('auto run = [&](int run_times = 1) -> double {\n')
    main_func_body += ('    t1 = std::chrono::high_resolution_clock::now();\n')
    main_func_body += ('    for (int i = 0; i < run_times; ++i) {\n')
    main_func_body += ('        results.clear();\n')
    main_func_body += ('        for (int j = 0; j < ' + str(rank) + '; ++j) {\n')
    main_func_body += ('            results.emplace_back(pool.enqueue(kernel_main');
    for i in range(len(inputs)):
        main_func_body += (', reinterpret_cast<{}*>(input{})'.format(input_types[i], str(i)))
    for i in range(len(outputs)):
        main_func_body += (', reinterpret_cast<{}*>(output{})'.format(output_types[i], str(i)))
    main_func_body += (', j));\n')
    main_func_body += ('        }\n')
    main_func_body += ('        for(auto && result: results) {\n')
    main_func_body += ('            result.get();\n')
    main_func_body += ('        }\n')
    main_func_body += ('    }\n')
    main_func_body += ('    t2 = std::chrono::high_resolution_clock::now();\n')
    main_func_body += ('    return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() / run_times;\n')
    main_func_body += ('};\n\n')

    # Warmup.
    main_func_body += ('// Warmup.\n')
    main_func_body += ('run(1);\n\n')

    # Measure step time.
    main_func_body += ('// Measure step time.\n')
    main_func_body += ('double sec;\n')
    main_func_body += ('sec = run(1);\n\n')
    
    # Adapt the test times.
    main_func_body += ('// Adapt the test times.\n')
    main_func_body += ('int test_run = 100;\n')
    main_func_body += ('if ( sec >= 1) test_run = 3;\n')
    main_func_body += ('if ( sec < 1 && sec >= 0.1) test_run = 10;\n')
    main_func_body += ('if ( sec < 0.1 && sec >= 0.01) test_run = 100;\n')
    main_func_body += ('if ( sec < 0.01 && sec >= 0.001) test_run = 1000;\n')
    main_func_body += ('if ( sec < 0.001 && sec >= 0.0001) test_run = 10000;\n')
    main_func_body += ('if ( sec < 0.0001) test_run = 100000;\n\n')

    # Test.
    main_func_body += ('// Testing.\n')
    main_func_body += ('sec = run(test_run);\n\n')

    main_func_body += ('double ans;\n')
    for i, output in enumerate(outputs):
        shape, type, name = output.split('/')
        dims = list(filter(None, shape.split('-')))
        size = 1
        for dim in dims:
            size *= int(dim)
        main_func_body += ('ans = 0;\n')
        main_func_body += ('for( int i = 0, ceof = 1; i < ' + str(size) + '; ++i, ceof = (ceof + 1) % 83) {\n')
        main_func_body += ('    ans += double(output' + str(i) + '[i]) * ceof;\n') 
        main_func_body += ('}\n')
        main_func_body += ('printf("- K/' + str(i) + ' = %.10e\\n", ans);\n')

    main_func_body += ('printf("- TPR = %.6e\\n", sec);\n')

    template = """
#include <iostream>
#include <vector>
#include <chrono>
#include <sys/resource.h>
#include "threadpool.h"


{kernel_func}

int main()
{{
{main_func_body}

return 0;
}}

""" 
    context = {
      "kernel_func": kernel_code, 
      "main_func_body": main_func_body
    }

    with open('main.cpp','w') as file:
        file.write(template.format(**context))
    
    return True, rank

def build_source_file():
    try:
        if platform.system() == 'Linux':
            cmd = ['g++', 'main.cpp', '-omain', '-std=c++11', '-lpthread', '-O3', '-march=native']
            output = subprocess.check_output(cmd, timeout=120)
        else:
            pass
    except CalledProcessError as e:
        print(e)
        return False

    return True

def execute(rank):
    mask = ""
    for i in range(rank):
        mask += "1"
    cmd = ["taskset", str(hex(int(mask, 2))), "./main"]
    logging.info("Starting execute: '%s'", ' '.join(cmd))
    output = b''
    try:
        output = subprocess.check_output(cmd, timeout=10)
    except subprocess.CalledProcessError as e:
        print(e)

    return output


def profile_kernel(kernel_source):
    ret, rank = generate_source_file(kernel_source)
    if ret == False:
        return False, 'parse kernel failed.'

    ret = build_source_file()
    if ret == False:
        return False, 'build kernel failed.'

    output = execute(rank)
    return True, output.decode('utf-8')

@tornado.web.stream_request_body
class PUTHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.bytes_read = 0
        
    def prepare(self):
        self.content = []

    def data_received(self, chunk):
        self.bytes_read += len(chunk)
        self.content.append(chunk.decode('utf-8'))

    def put(self, filename):
        mtype = self.request.headers.get("Content-Type")
        logging.info('PUT "%s" "%s" %d bytes', filename, mtype, self.bytes_read)

        ret, output = profile_kernel(''.join(self.content))
        
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
