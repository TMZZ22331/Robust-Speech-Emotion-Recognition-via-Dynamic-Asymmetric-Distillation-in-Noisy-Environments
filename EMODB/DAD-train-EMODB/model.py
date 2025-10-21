import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Emotion2Vec基础模型 ---
class Emotion2VecEncoder(nn.Module):
    """基于emotion2vec的编码器模块"""
    
    def __init__(self, input_dim=768, hidden_dim=256, pretrained_path=None):
        super().__init__()
        
        # 预网络：将emotion2vec特征映射到隐藏维度
        self.pre_net = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.activate = nn.ReLU()
        
        print(f"🏗️ Emotion2Vec编码器: {input_dim} → {hidden_dim}")
    
    def forward(self, x, padding_mask=None):
        """
        前向传播
        Args:
            x: (batch_size, seq_len, input_dim) - emotion2vec特征
            padding_mask: (batch_size, seq_len) - 填充掩码，True表示填充位置
        Returns:
            encoded_features: (batch_size, hidden_dim) - 编码后的特征
        """
        # 通过预网络
        x = self.activate(self.pre_net(x))  # (B, T, hidden_dim)
        
        # 处理padding mask并进行平均池化
        if padding_mask is not None:
            # 将padding位置置零
            x = x * (1 - padding_mask.unsqueeze(-1).float())
            # 计算有效长度的平均值
            valid_lengths = (1 - padding_mask.float()).sum(dim=1, keepdim=True)  # (B, 1)
            x = x.sum(dim=1) / torch.clamp(valid_lengths, min=1.0)  # (B, hidden_dim)
        else:
            # 无mask时直接平均池化
            x = x.mean(dim=1)  # (B, hidden_dim)
            
        return x

# --- 分类器模块 ---
class EmotionClassifier(nn.Module):
    """情绪分类器"""
    
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        
        # 添加dropout提高泛化能力
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_layer = nn.Linear(in_features=input_dim, out_features=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: (batch_size, input_dim) - 编码器输出的特征
        Returns:
            logits: (batch_size, num_classes) - 分类logits
        """
        x = self.dropout(x)
        logits = self.fc_layer(x)
        return logits

# --- SSRL主框架 ---
class SSRLModel(nn.Module):
    """
    完整的 Self-Supervised Reflective Learning (SSRL) 架构。
    
    整合了：
    1. Emotion2Vec编码器作为backbone
    2. KL散度约束（教师-学生一致性）
    3. 监督对比损失（SCL）- 类别对齐
    4. 最大均值差异（MMD）- 域对齐
    5. Top-K动态置信度筛选
    """
    def __init__(self, cfg):
        """
        初始化完整的SSRL模型。
        
        参数:
            cfg: 配置对象，需要包含：
                - INPUT_DIM: emotion2vec特征维度 (通常是768)
                - HIDDEN_DIM: 编码器隐藏层维度 (默认256)
                - NUM_CLASSES: 分类类别数 (通常是4: ang, hap, neu, sad)
                - EMA_MOMENTUM: 教师网络EMA更新动量
                - DROPOUT_RATE: 分类器dropout率
                - PRETRAINED_EMOTION2VEC_PATH: 预训练模型路径
        """
        super().__init__()
        
        # 从配置中获取参数
        input_dim = getattr(cfg, 'INPUT_DIM', 768)
        hidden_dim = getattr(cfg, 'HIDDEN_DIM', 256) 
        num_classes = getattr(cfg, 'NUM_CLASSES', 4)
        dropout_rate = getattr(cfg, 'DROPOUT_RATE', 0.1)
        pretrained_path = getattr(cfg, 'PRETRAINED_EMOTION2VEC_PATH', None)
        
        # --- 1. 创建学生网络 ---
        self.student_encoder = Emotion2VecEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pretrained_path=None  # 先不加载，后面统一加载
        )
        self.student_classifier = EmotionClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        
        # --- 2. 创建教师网络 ---
        self.teacher_encoder = Emotion2VecEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pretrained_path=None  # 教师网络通过EMA更新，不直接加载预训练权重
        )
        self.teacher_classifier = EmotionClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=0.0  # 教师网络在推理时不使用dropout
        )
        
        # --- 3. 加载完整的预训练权重（编码器+分类器） ---
        if pretrained_path:
            self.load_complete_pretrained_weights(pretrained_path)
        
        # --- 4. 初始化教师网络 ---
        # 将教师网络的权重初始化为与学生网络完全相同
        # 并将其设置为不计算梯度，因为教师是通过EMA更新的
        self._init_teacher_network()
            
        # 存储EMA动量超参数
        self.ema_momentum = getattr(cfg, 'EMA_MOMENTUM', 0.99)
        
        print(f"🚀 完整SSRL模型初始化完成")
        print(f"📊 输入维度: {input_dim}, 隐藏维度: {hidden_dim}, 类别数: {num_classes}")
        print(f"⚖️ EMA动量: {self.ema_momentum}")
        print(f"🎯 集成组件: Emotion2Vec + KL散度 + SCL损失 + MMD损失")
        if pretrained_path:
            print(f"🎯 预训练路径: {pretrained_path}")

    def load_complete_pretrained_weights(self, pretrained_path):
        """
        加载完整的预训练权重（编码器+分类器）
        从你训练好的emotion2vec模型中加载pre_net和post_net权重
        """
        try:
            print(f"🔄 正在加载完整预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            
            # 权重文件直接包含张量，不需要嵌套字典
            print(f"📂 发现权重键: {list(checkpoint.keys())}")
            
            # 统计加载情况
            encoder_loaded = 0
            classifier_loaded = 0
            
            # === 1. 加载编码器权重 (pre_net) ===
            encoder_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('pre_net'):
                    encoder_state_dict[key] = value
                    encoder_loaded += 1
            
            if encoder_state_dict:
                self.student_encoder.load_state_dict(encoder_state_dict, strict=False)
                print(f"✅ 编码器权重加载成功: {encoder_loaded} 个参数")
            
            # === 2. 加载分类器权重 (post_net -> fc_layer) ===
            classifier_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('post_net'):
                    # 将post_net映射到fc_layer
                    new_key = key.replace('post_net', 'fc_layer')
                    classifier_state_dict[new_key] = value
                    classifier_loaded += 1
            
            if classifier_state_dict:
                self.student_classifier.load_state_dict(classifier_state_dict, strict=False)
                print(f"✅ 分类器权重加载成功: {classifier_loaded} 个参数")
            
            # === 3. 总结 ===
            total_loaded = encoder_loaded + classifier_loaded
            print(f"🎯 预训练权重加载完成: 总共 {total_loaded} 个参数")
            print(f"   📊 编码器(pre_net): {encoder_loaded} 个参数")
            print(f"   🎯 分类器(post_net): {classifier_loaded} 个参数")
            
            if total_loaded == 4:  # 2个编码器 + 2个分类器
                print(f"✅ 权重加载完整！模型已使用你训练好的分类头")
            else:
                print(f"⚠️ 权重加载不完整，部分使用随机初始化")
            
        except Exception as e:
            print(f"❌ 加载预训练权重失败: {e}")
            print("🔄 将使用随机初始化权重")
            import traceback
            traceback.print_exc()

    def _init_teacher_network(self):
        """初始化教师网络权重并设置为不需要梯度"""
        # 复制学生网络权重到教师网络
        for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.student_encoder.parameters()):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False
            
        for teacher_param, student_param in zip(self.teacher_classifier.parameters(), self.student_classifier.parameters()):
            teacher_param.data.copy_(student_param.data)
            teacher_param.requires_grad = False

    @torch.no_grad()
    def update_teacher_ema(self):
        """
        使用学生网络的权重，通过指数移动平均（EMA）来更新教师网络。
        这个函数应该在每次学生网络权重更新（optimizer.step()）之后调用。
        """
        # 更新编码器
        for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.student_encoder.parameters()):
            teacher_param.data = teacher_param.data * self.ema_momentum + student_param.data * (1.0 - self.ema_momentum)
            
        # 更新分类器
        for teacher_param, student_param in zip(self.teacher_classifier.parameters(), self.student_classifier.parameters()):
            teacher_param.data = teacher_param.data * self.ema_momentum + student_param.data * (1.0 - self.ema_momentum)

    def predict(self, x: torch.Tensor, padding_mask=None, use_teacher=False):
        """
        预测函数，用于推理阶段
        
        参数:
            x: 输入特征
            padding_mask: 填充掩码  
            use_teacher: 是否使用教师网络进行预测
            
        返回:
            logits: 预测logits
        """
        self.eval()
        with torch.no_grad():
            if use_teacher:
                embedding = self.teacher_encoder(x, padding_mask)
                logits = self.teacher_classifier(embedding)
            else:
                embedding = self.student_encoder(x, padding_mask)
                logits = self.student_classifier(embedding)
        return logits

    def get_embeddings(self, x: torch.Tensor, padding_mask=None, use_teacher=False):
        """
        获取编码器输出的特征嵌入
        
        参数:
            x: 输入特征
            padding_mask: 填充掩码
            use_teacher: 是否使用教师网络
            
        返回:
            embeddings: 特征嵌入
        """
        self.eval()
        with torch.no_grad():
            if use_teacher:
                embeddings = self.teacher_encoder(x, padding_mask)
            else:
                embeddings = self.student_encoder(x, padding_mask)
        return embeddings