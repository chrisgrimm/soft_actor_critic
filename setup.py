#! /usr/bin/env python

from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='soft-actor-critic',
      version='0.0.0',
      description='An implementation of Soft Actor Critic',
      long_description=long_description,
      url='https://github.com/chrisgrimm/soft_actor_critic',
      author='Chris Grimm',
      author_email='cgrimm1994@gmail.com',
      packages=['runs'],
      install_requires=[
          'paramiko==2.3.1',
          'termcolor==1.1.0',
          'PyYAML==3.12',
          'tabulate==0.8.1',
      ])
