#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def chdir(dir_, logger, create=False):
    if not dir_.is_dir():
        if create:
            logger.warning(f"The directory {dir_} doesn't exist. Creating it")
            dir_.mkdir(parents=True, exist_ok=True)
        else:
            logger.error(f"The directory {dir_} doesn't exist")
            raise SystemExit(1)
