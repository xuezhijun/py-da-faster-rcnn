# -*- coding: utf-8 -*-
from xml.dom.minidom import Document


def writexml(filename, img_shape, bboxes, categories, instance_semantics, xmlpath):
    """
    write to xml style of VOC dataset
    :param filename: xml filename
    :param img_shape: the image shape [H, W, C]
    :param bboxes: bounding boxes
    :param categories: category id
    :param instance_semantics: category id corresponding semantic
    :param xmlpath: xml file save path
    :return: None
    """
    doc = Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('Cityscapes')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filenamenode = doc.createElement('filename')
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)

    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('Cityscapes'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    owner = doc.createElement('owner')
    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('kinredon'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(img_shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(img_shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(img_shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(instance_semantics[categories[i] - 1]))
        objects.appendChild(object_name)

        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[3])))
        bndbox.appendChild(ymax)

    with open(xmlpath, 'w') as f:
        f.write(doc.toprettyxml(indent=' '))
