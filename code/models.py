import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class InstanceFeatureExtractor(nn.Module):
    """Extract features from individual patches"""
    def __init__(self, pretrained=True):
        super(InstanceFeatureExtractor, self).__init__()
        
        # ResNet34 backbone for feature extraction (lighter than ResNet50)
        resnet = models.resnet34(pretrained=pretrained)
        
        # Remove final classification layers
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.feature_dim = 512  # ResNet34 feature dimension
        
    def forward(self, x):
        """
        x: [batch_size * num_patches, 3, 512, 512]
        returns: [batch_size * num_patches, feature_dim]
        """
        features = self.features(x)  # [N, 512, 16, 16]
        pooled = self.gap(features)  # [N, 512, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [N, 512]
        return pooled

class AttentionMILPooling(nn.Module):
    """Attention-based MIL aggregation"""
    def __init__(self, feature_dim=512, hidden_dim=256):
        super(AttentionMILPooling, self).__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, instance_features, bag_sizes):
        """
        instance_features: [total_instances, feature_dim] - all instances from all bags
        bag_sizes: [batch_size] - number of instances in each bag
        returns: [batch_size, feature_dim] - aggregated bag representations
        """
        # Compute attention weights for all instances
        attention_weights = self.attention(instance_features)  # [total_instances, 1]
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        # Split by bags and aggregate
        bag_representations = []
        start_idx = 0
        
        for bag_size in bag_sizes:
            end_idx = start_idx + bag_size
            
            # Get instances and weights for this bag
            bag_instances = instance_features[start_idx:end_idx]  # [bag_size, feature_dim]
            bag_weights = attention_weights[start_idx:end_idx]  # [bag_size, 1]
            
            # Weighted aggregation
            bag_repr = torch.sum(bag_instances * bag_weights, dim=0)  # [feature_dim]
            bag_representations.append(bag_repr)
            
            start_idx = end_idx
        
        return torch.stack(bag_representations)  # [batch_size, feature_dim]

class LearnableThresholds(nn.Module):
    def __init__(self, num_thresholds=1, init_values=None):
        super(LearnableThresholds, self).__init__()
        
        if init_values is None:
            init_values = torch.linspace(0.3, 0.7, num_thresholds)
        
        self.thresholds = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))
        
    def forward(self, scores):
        sorted_thresholds = torch.sort(self.thresholds)[0]
        
        grades = torch.zeros_like(scores, dtype=torch.long)
        for i, threshold in enumerate(sorted_thresholds):
            grades = torch.where(scores > threshold, i + 1, grades)
            
        return grades, sorted_thresholds

class LRRC15GradePredictor(nn.Module):
    """True Multiple Instance Learning model for LRRC15 grade prediction"""
    def __init__(self, num_grades=2, pretrained=True, use_attention=True, learnable_thresholds=True):
        super(LRRC15GradePredictor, self).__init__()
        
        # Instance-level feature extractor
        self.instance_extractor = InstanceFeatureExtractor(pretrained=pretrained)
        feature_dim = self.instance_extractor.feature_dim
        
        # MIL aggregation
        self.use_attention = use_attention
        if use_attention:
            self.mil_pooling = AttentionMILPooling(feature_dim=feature_dim)
        else:
            # Simple mean pooling as baseline
            self.mil_pooling = None
        
        # H-score prediction head
        self.h_score_predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Grade classification head
        self.grade_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_grades)
        )
        
        # Learnable thresholds
        self.learnable_thresholds = learnable_thresholds
        if learnable_thresholds:
            self.threshold_module = LearnableThresholds(num_thresholds=num_grades-1)
        
    def forward(self, bags, bag_sizes):
        """
        bags: [total_instances, 3, 512, 512] - all patches from all bags in batch
        bag_sizes: [batch_size] - number of patches in each bag
        """
        # Extract features from all instances
        instance_features = self.instance_extractor(bags)  # [total_instances, feature_dim]
        
        # MIL aggregation
        if self.use_attention:
            bag_features = self.mil_pooling(instance_features, bag_sizes)
        else:
            # Simple mean pooling
            bag_features = []
            start_idx = 0
            for bag_size in bag_sizes:
                end_idx = start_idx + bag_size
                bag_feat = torch.mean(instance_features[start_idx:end_idx], dim=0)
                bag_features.append(bag_feat)
                start_idx = end_idx
            bag_features = torch.stack(bag_features)
        
        # Predictions
        h_score_norm = self.h_score_predictor(bag_features).squeeze()
        grade_logits = self.grade_classifier(bag_features)
        
        # Threshold-based grades
        if self.learnable_thresholds:
            grade_from_threshold, thresholds = self.threshold_module(h_score_norm)
        else:
            grade_from_threshold = None
            thresholds = None
        
        return {
            'h_score_norm': h_score_norm,
            'grade_from_threshold': grade_from_threshold,
            'grade_logits': grade_logits,
            'thresholds': thresholds,
            'bag_features': bag_features,
            'instance_features': instance_features
        }