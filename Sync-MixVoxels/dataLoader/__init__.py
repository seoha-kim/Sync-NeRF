from .llff_video import LLFFVideoDataset, SSDDataset
from .blender_video import BlenderVideoDataset
dataset_dict = {'ssd': SSDDataset, 'llffvideo':LLFFVideoDataset, 'blendervideo':BlenderVideoDataset}