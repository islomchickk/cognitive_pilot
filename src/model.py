import torch
import torch.nn as nn


class RotateLayer(nn.Module):
    def __init__(self, initial_angle=0.0):
        super(RotateLayer, self).__init__()
        # Инициализируем угол поворота как обучаемый параметр
        self.theta = nn.Parameter(torch.tensor(initial_angle))
        
    def get_rotation_matrix(self, theta):
        # Создаем матрицу поворота 2x2
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
                
        # Собираем матрицу поворота, сохраняя вычислительный граф
        return torch.stack([
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta, cos_theta])
        ])
    
    def forward(self, x):

        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        
        # Создаем сетку координат
        x_coords = torch.linspace(-1, 1, width, requires_grad=False).to(x.device)
        y_coords = torch.linspace(-1, 1, height, requires_grad=False).to(x.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Объединяем координаты в тензор
        grid = torch.stack([grid_x, grid_y], dim=-1).to(x.device)
        
        # Получаем матрицу поворота
        rot_matrix = self.get_rotation_matrix(self.theta)
        
        # Применяем поворот к сетке координат
        grid = grid.view(-1, 2)
        rotated_grid = torch.matmul(grid.to(x.device), rot_matrix.T.to(x.device))
        rotated_grid = rotated_grid.view(height, width, 2)
        
        # Преобразуем сетку в формат для grid_sample
        grid_sample = rotated_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Применяем sampling
        rotated_images = torch.nn.functional.grid_sample(
            x, grid_sample, align_corners=True, mode='bilinear'
        )
        return rotated_images