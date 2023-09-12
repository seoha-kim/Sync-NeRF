from .llff_video import LLFFVideoDataset, SSDDataset
from .blender_video import BlenderVideoDataset

dataset_dict = {'blendervideo':BlenderVideoDataset,'ssd': SSDDataset, 'llffvideo':LLFFVideoDataset}