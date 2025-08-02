import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import timm


def _weights_init(m):
    classname = m.__class__.__name__
        #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Mish(nn.Module):
    def __init__(self, lambd=0.0):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = Mish()


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                    """
                    For CIFAR10 ResNet paper uses option A.
                    """
                    self.shortcut = LambdaLayer(lambda x:
                                                F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):

        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.act = Mish()

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


class ImageClassificationBase(nn.Module):
    
    '''
    Abstract class that extends nn.Module and adds some functions that make training a lot nicer. 
    The functions provide the steps taken when training and validating the model, as well as code 
    to print the loss and accuracy at each epoch. 
    '''
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        acc = accuracy(out, labels)
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['train_acc'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class EfficientNetv2s_pre_yes(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', out_dim=256, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class EfficientNetv2s_pre_no(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s', out_dim=256, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class EfficientNetB4_NS_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b4.ns_jft_in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class EfficientNetB4_NS_pre_no(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b4.ns_jft_in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class EfficientNetB4_ORG_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b4', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class EfficientNetB4_ORG_pre_no(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b4', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class EfficientNetB1_ORG_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b1.in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class EfficientNetB1_ORG_pre_no(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b1.in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


# prompt: modify EfficientNetB0_NS_pre_yes by adding a new fully connected layer including 512 features and fc1 add dropout
class EfficientNetB0_NS_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b0.ns_jft_in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(n_features, 512)
        self.dropout = nn.Dropout(0.3) # Added dropout layer
        self.classifier = nn.Linear(512, out_dim) # Modified classifier

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        x = self.fc1(pooled_features) # Added fc1 layer
        x = self.dropout(x) # Added dropout
        output = self.classifier(x) # Modified classifier
        return output


class EfficientNetB0_NS_pre_no(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b0.ns_jft_in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class EfficientNetB2_NS_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b2.ns_jft_in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class EfficientNetB2_NS_pre_no(ImageClassificationBase):
    def __init__(self, model_name='tf_efficientnet_b2.ns_jft_in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class ResNet18_pre_yes(nn.Module):
    def __init__(self, model_name='resnet18', out_dim=256, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        
        #if pretrained:
        #    pretrained_path = '/content/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
           
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)
       

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output

class ResNet18_pre_no(nn.Module):
    def __init__(self, model_name='resnet18', out_dim=256, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        
        #if pretrained:
        #    pretrained_path = '/content/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
           
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, out_dim)
       

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output



class DLA60X_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='dla60x_c.in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.num_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        #pooled_features = self.pooling(features)
        output = self.classifier(features)
        return output


class DLA60X_pre_no(ImageClassificationBase):
    def __init__(self, model_name='dla60x_c.in1k', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.num_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        #pooled_features = self.pooling(features)
        output = self.classifier(features)
        return output


class MobileNet_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='mobilenetv2_050', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class MobileNet_pre_no(ImageClassificationBase):
    def __init__(self, model_name='mobilenetv2_050', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

class TinyNet_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='tinynet_c', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class TinyNet_pre_no(ImageClassificationBase):
    def __init__(self, model_name='tinynet_c', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

        
class Hardcorenas_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='hardcorenas_d', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class Hardcorenas_pre_no(ImageClassificationBase):
    def __init__(self, model_name='hardcorenas_d', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class XcitTinyNet_pre_yes(ImageClassificationBase):
    def __init__(self, model_name='xcit_tiny_12_p8_384', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output


class XcitTinyNet_pre_no(ImageClassificationBase):
    def __init__(self, model_name='xcit_tiny_12_p8_384', out_dim=256, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        n_features = self.model.classifier.in_features
        self.model.global_pool = nn.Identity() #global pool olan yeri iptal eder direk gecis olur
        self.model.classifier = nn.Identity() #classifer iptal eder yani 1x1000 ortadan kalkar direk gecis olur
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_features, out_dim)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.classifier(pooled_features)
        return output

