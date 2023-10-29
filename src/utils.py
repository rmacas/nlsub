#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def chdir(dir_, logger):
    if not dir_.is_dir():
        logger.error(f"The directory {dir_} doesn't exist")
        raise SystemExit(1)
