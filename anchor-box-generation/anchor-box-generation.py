import math

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    anchors = []
    # 1. Compute the stride
    stride = image_size / feature_size
    
    # 2. Iterate over the grid in row-major order (i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            # Compute center of the current cell in image coordinates
            cy = (i + 0.5) * stride
            cx = (j + 0.5) * stride
            
            # 3. For each cell, iterate over scales then aspect ratios
            for s in scales:
                for r in aspect_ratios:
                    # Compute width and height based on scale and ratio
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    # 4. Create the box [x1, y1, x2, y2]
                    anchor = [
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2
                    ]
                    anchors.append(anchor)
                    
    return anchors
