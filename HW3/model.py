import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead


def get_model(model_name="maskrcnn_resnet50_fpn_v2"):
    if model_name == "maskrcnn_resnet50_fpn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT"
        )

    num_classes = 5  # background + 4 classes

    # box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    # mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(
        model.backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
    )

    return model
