import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours

def all_black_colors(N):
    """
    Generate all black colors
    """
    black = [(0, 0, 0) for i in range(N)]
    return black

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def create_mask(image, boxes, masks, class_ids,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=False, colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        print("detected")
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        auto_show = False

    # Generate black colors
    colors = colors or all_black_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]


    masked_image = image.astype(np.uint32).copy()
    mask_white = np.full((height, width, 3), 255, dtype=int)
    
    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[:, :, i]
        mask_white = apply_mask(mask_white, mask, color, alpha=1)

    return mask_white

def test_create_mask(image, boxes, masks, class_ids,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=False, colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        print("detected")
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = False

    # Generate random colors
    #colors = colors or random_colors(N)
    colors = colors or all_black_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    mask_white = np.full((height, width, 3), 255, dtype=int)
    
    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            mask_white = apply_mask(mask_white, mask, color, alpha=1)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(mask_white.astype(np.uint8))
    if auto_show:
        plt.show()
    return mask_white
