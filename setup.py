from setuptools import find_packages, setup

setup(
    name="probing-vlm-vgm",
    version="1.0",
    description="Frozen-feature probing for VLM and VGM spatial intelligence",
    author="Haozhan Shen",
    packages=find_packages(include=["probing_vlm_vgm", "probing_vlm_vgm.*"]),
)
