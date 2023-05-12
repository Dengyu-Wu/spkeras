# Build the pip installable package

PKG_SRC_DIR=./spkeras
PKG_NAME=spkeras
PKG_AUTHOR=""
PKG_AUTHOR_EMAIL=""
PKG_DESC="Spiking Neural Networks in Keras"
PKG_LONG_DESC="Spiking Neural Networks in Keras"
PKG_REQUIRES="\'tensorflow\', \'numpy\'"
PKG_LICENSE="MIT"
PKG_URL="https://github.com/Dengyu-Wu/spkeras.git"
PKG_VERSION=2.0.1

SRC_FILES=$(shell find $(PKG_SRC_DIR) -type f -name "*.py")

BUILD_DIR=./build

.PHONY: all clean install uninstall

all: $(SRC_FILES)
	mkdir -p $(BUILD_DIR)
	cp -r $(PKG_SRC_DIR) $(BUILD_DIR)/$(PKG_NAME)

	# Generate setup.py
	echo "from setuptools import setup, find_packages" > $(BUILD_DIR)/setup.py
	echo "setup(" >> $(BUILD_DIR)/setup.py
	echo "    name=\"$(PKG_NAME)\"," >> $(BUILD_DIR)/setup.py
	echo "    version=\"$(PKG_VERSION)\"," >> $(BUILD_DIR)/setup.py
	echo "    author=\"$(PKG_AUTHOR)\"," >> $(BUILD_DIR)/setup.py
	echo "    author_email=\"$(PKG_AUTHOR_EMAIL)\"," >> $(BUILD_DIR)/setup.py
	echo "    description=\"$(PKG_DESC)\"," >> $(BUILD_DIR)/setup.py
	echo "    long_description=\"$(PKG_LONG_DESC)\"," >> $(BUILD_DIR)/setup.py
	echo "    long_description_content_type=\"text/markdown\"," >> $(BUILD_DIR)/setup.py
	echo "    license=\"$(PKG_LICENSE)\"," >> $(BUILD_DIR)/setup.py
	echo "    url=\"$(PKG_URL)\"," >> $(BUILD_DIR)/setup.py
	echo "    packages=find_packages()," >> $(BUILD_DIR)/setup.py
	echo "    install_requires=[$(PKG_REQUIRES)]," >> $(BUILD_DIR)/setup.py
	echo "    classifiers=[" >> $(BUILD_DIR)/setup.py
	echo "        \"Programming Language :: Python :: 3\"," >> $(BUILD_DIR)/setup.py
	echo "        \"License :: OSI Approved :: MIT License\"," >> $(BUILD_DIR)/setup.py
	echo "        \"Operating System :: OS Independent\"," >> $(BUILD_DIR)/setup.py
	echo "    ]," >> $(BUILD_DIR)/setup.py
	echo ")" >> $(BUILD_DIR)/setup.py

	# Generate MANIFEST.in
	echo "include README.md" > $(BUILD_DIR)/MANIFEST.in
	echo "include LICENSE" >> $(BUILD_DIR)/MANIFEST.in

	# Generate LICENSE
	cp LICENSE $(BUILD_DIR)/LICENSE

	# Generate README.md
	cp README.md $(BUILD_DIR)/README.md

	# Build the package
	cd $(BUILD_DIR) && python setup.py sdist bdist_wheel --universal

clean:
	rm -rf $(BUILD_DIR)

install:
	pip install $(BUILD_DIR)/dist/$(PKG_NAME)-$(PKG_VERSION).tar.gz

uninstall:
	pip uninstall $(PKG_NAME)