from glob import glob
import os.path as osp

from setuptools import find_packages
from setuptools import setup

package_name = "autoware_mtr_python"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(include=["autoware_mtr*", "autoware_mtr_python*","awml_pred*","projects*"],exclude=["test"]),
    data_files=[
        (osp.join("share", package_name), ["package.xml"]),
        (osp.join("share", package_name, "config"), glob("config/*")),
        (osp.join("share", package_name, "launch"), glob("launch/*")),
        (osp.join("share", package_name, "data"), glob("data/*")),
        (osp.join("share", package_name, "src"), glob("src/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="danielsanchezaran",
    maintainer_email="daniel.sanchez@tier4.jp",
    description="A ROS package of MTR written in Python.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"{package_name}_node = src.mtr_node:main",
        ],
    },
)
