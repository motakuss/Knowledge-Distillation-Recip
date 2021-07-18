import torch
import torch.nn as nn

class TeacherConvBnAct(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.dropout = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class StudentConvBnAct(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def _make_teacher_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(TeacherConvBnAct(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)
    
    def _make_student_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(StudentConvBnAct(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)

# 16-layer model
class StudentModel(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_student_layer(4, input_dim, 16)
        self.layer2 = self._make_student_layer(4, 16, 32)
        self.layer3 = self._make_student_layer(4, 32, 64)
        self.layer4 = self._make_student_layer(4, 64, 128)
        self.mlp = MLP(128, 128, 10)
        self.maxpool = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
    def extract_features(self,x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        return x
    
# 12-layer model
class TeacherAssistantModel(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_student_layer(3, input_dim, 32)
        self.layer2 = self._make_student_layer(3, 32, 64)
        self.layer3 = self._make_student_layer(3, 64, 128)
        self.layer4 = self._make_student_layer(3, 128, 256)
        self.mlp = MLP(256, 128, 10)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
    def extract_features(self,x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        return x
        
# 8-layer model
class TeacherModel(BaseModel):
    def __init__(self, input_dim=3):
        super().__init__()
        self.layer1 = self._make_teacher_layer(2, input_dim, 64)
        self.layer2 = self._make_teacher_layer(2, 64, 128)
        self.layer3 = self._make_teacher_layer(2, 128, 256)
        self.layer4 = self._make_teacher_layer(2, 256, 512)
        self.mlp = MLP(512, 128, 10)
        self.maxpool = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x
    
    def extract_features(self,x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        return x
    
class BasicBlock(nn.Module):
    rate=1
    def __init__(self, input_dim, output_dim,stride=1,drop=0.3):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(drop),
            self.relu,
            nn.Conv2d(output_dim, output_dim * BasicBlock.rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim * BasicBlock.rate),
            nn.Dropout(drop)
        )
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or input_dim != output_dim * BasicBlock.rate:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim * BasicBlock.rate, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_dim * BasicBlock.rate)
            )
    
    def forward(self, x):
        return self.relu(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    rate = 4
    def __init__(self, input_dim, output_dim, stride=1, drop=0.3):
        super().__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(drop),
            self.relu,
            nn.Conv2d(output_dim, output_dim, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.Dropout(drop),
            self.relu,
            nn.Conv2d(output_dim, output_dim * BottleNeck.rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim * BottleNeck.rate),
            nn.Dropout(drop),
        )
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or input_dim != output_dim * BottleNeck.rate:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim * BottleNeck.rate, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_dim * BottleNeck.rate)
            )
    
    def forward(self, x):
        return self.relu((self.residual_function(x) + self.shortcut(x)))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100, drop=0.3):
        super().__init__()
        
        self.input_dim = 64
        self.drop = drop
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(512 * block.rate, num_classes)
    
    def _make_layer(self, block, output_dim, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.input_dim, output_dim, stride,self.drop))
            self.input_dim = output_dim * block.rate
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.avg_pool(x)
        return x
        
    
def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])

def resnet101():
    return ResNet(BottleNeck, [3,4,23,3])

def resnet152():
    return ResNet(BottleNeck, [3,8,36,3])

if __name__ == '__main__':
    x = torch.randn((4, 3, 32, 32))
    teacher = TeacherModel()
    student = StudentModel()
    ta = TeacherAssistantModel()
    teacher_output = teacher(x)
    student_output = student(x)
    ta_output = ta(x)
    print(teacher_output.shape)
    print(student_output.shape)
    print(ta_output.shape)
    print(teacher_output)
    print(student_output)
    print(ta_output)
        
