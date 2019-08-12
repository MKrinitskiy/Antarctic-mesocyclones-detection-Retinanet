#!/usr/bin/env python
# -*- coding: utf8 -*-
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import numpy as np

PASCAL_XML_EXT = '.PASCAL.xml'
MCC_XML_EXT = '.MCC.xml'
ENCODE_METHOD = 'utf-8'

class ArbitraryXMLWriter:

    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        return top

    def addEllipse(self, ellipse_shape):
        ellipse_properties = {'lon0': ellipse_shape.latlonPoints[0].x(),
                              'lat0': ellipse_shape.latlonPoints[0].y(),
                              'lon1': ellipse_shape.latlonPoints[1].x(),
                              'lat1': ellipse_shape.latlonPoints[1].y(),
                              'lon2': ellipse_shape.latlonPoints[2].x(),
                              'lat2': ellipse_shape.latlonPoints[2].y()}
        ellipse_properties['name'] = ellipse_shape.label
        ellipse_properties['isEllipse'] = ellipse_shape.isEllipse
        self.boxlist.append(ellipse_properties)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            try:
                name.text = unicode(each_object['name'])
            except NameError:
                # Py3: NameError: name 'unicode' is not defined
                name.text = each_object['name']

            ellipse = SubElement(object_item, 'ellipse')

            # x0 = SubElement(ellipse, 'x0')
            # x0.text = str(each_object['x0'])
            # y0 = SubElement(ellipse, 'y0')
            # y0.text = str(each_object['y0'])
            # x1 = SubElement(ellipse, 'x1')
            # x1.text = str(each_object['x1'])
            # y1 = SubElement(ellipse, 'y1')
            # y1.text = str(each_object['y1'])
            # x2 = SubElement(ellipse, 'x2')
            # x2.text = str(each_object['x2'])
            # y2 = SubElement(ellipse, 'y2')
            # y2.text = str(each_object['y2'])

            lon0 = SubElement(ellipse, 'lon0')
            lon0.text = str(each_object['lon0'])
            lat0 = SubElement(ellipse, 'lat0')
            lat0.text = str(each_object['lat0'])
            lon1 = SubElement(ellipse, 'lon1')
            lon1.text = str(each_object['lon1'])
            lat1 = SubElement(ellipse, 'lat1')
            lat1.text = str(each_object['lat1'])
            lon2 = SubElement(ellipse, 'lon2')
            lon2.text = str(each_object['lon2'])
            lat2 = SubElement(ellipse, 'lat2')
            lat2.text = str(each_object['lat2'])

            isEllipse = SubElement(ellipse, 'isEllipse')
            isEllipse.text = str(each_object['isEllipse'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + MCC_XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()


class ArbitraryXMLReader:

    def __init__(self, filepath):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color]
        self.shapes = []
        self.filepath = filepath
        self.verified = False
        try:
            self.parseXML()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, bndbox):
        # x0 = float(bndbox.find('x0').text)
        # y0 = float(bndbox.find('y0').text)
        # x1 = float(bndbox.find('x1').text)
        # y1 = float(bndbox.find('y1').text)
        # x2 = float(bndbox.find('x2').text)
        # y2 = float(bndbox.find('y2').text)

        lon0 = float(bndbox.find('lon0').text)
        lat0 = float(bndbox.find('lat0').text)
        lon1 = float(bndbox.find('lon1').text)
        lat1 = float(bndbox.find('lat1').text)
        lon2 = float(bndbox.find('lon2').text)
        lat2 = float(bndbox.find('lat2').text)

        # points = [(x0, y0), (x1, y1), (x2, y2)]
        latlonPoints = [(lon0, lat0), (lon1, lat1), (lon2, lat2)]
        isEllipse = True if bndbox.find('isEllipse').text.lower() == 'true' else False
        # self.shapes.append((label, points, latlonPoints, None, None, isEllipse))
        self.shapes.append((label, latlonPoints, None, None, isEllipse))

    def parseXML(self):
        assert self.filepath.endswith(MCC_XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.filepath, parser=parser).getroot()
        filename = xmltree.find('filename').text

        for object_iter in xmltree.findall('object'):
            ellipse = object_iter.find("ellipse")
            label = object_iter.find('name').text
            # Add chris
            self.addShape(label, ellipse)
        return True




def label_xmlfile_contents(fname):
    tree = etree.parse(fname)
    filename_element = tree.find('filename')
    retdict = {'nc_fname': filename_element.text}

    objects = []
    for object_elem in tree.findall('object'):
        t2 = etree.ElementTree(element=object_elem)
        curr_object = {'type': t2.find('name').text}
        for object_elem in t2.getiterator():
            curr_object['lat0'] = np.float64(t2.find('ellipse/lat0').text)
            curr_object['lon0'] = np.float64(t2.find('ellipse/lon0').text)
            curr_object['lat1'] = np.float64(t2.find('ellipse/lat1').text)
            curr_object['lon1'] = np.float64(t2.find('ellipse/lon1').text)
            curr_object['lat2'] = np.float64(t2.find('ellipse/lat2').text)
            curr_object['lon2'] = np.float64(t2.find('ellipse/lon2').text)

        objects.append(curr_object)

    retdict['objects'] = objects
    return retdict