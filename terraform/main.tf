# ============================================================================
# Main — Recursos de infraestrutura na AWS
# ============================================================================
#
# Recursos criados:
#   1. ECR         - Repositorio de imagens Docker (Docker Hub privado da AWS)
#   2. VPC         - Rede privada na AWS
#   3. Subnets     - Sub-redes publicas em 2 AZs (alta disponibilidade)
#   4. Internet GW - Porta de saida para a internet
#   5. Security Gr - Firewall (quais portas podem receber trafego)
#   6. ALB         - Load Balancer (distribui requisicoes entre containers)
#   7. ECS Cluster - Cluster de containers
#   8. Task Def    - Definicao do container (imagem, CPU, RAM, portas)
#   9. ECS Service - Garante que N containers estejam sempre rodando
#  10. IAM Roles   - Permissoes para o ECS acessar ECR e CloudWatch
#  11. CloudWatch  - Logs centralizados
# ============================================================================

# ── Provider ─────────────────────────────────────────────────────────────────
# Configura o Terraform para usar a AWS na regiao definida nas variaveis.
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # Tags padrao aplicadas em todos os recursos automaticamente
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Data Sources ─────────────────────────────────────────────────────────────
# Busca as Availability Zones disponiveis na regiao (ex: us-east-1a, us-east-1b)
data "aws_availability_zones" "available" {
  state = "available"
}

# ============================================================================
# 1. ECR — Elastic Container Registry
# ============================================================================
# Repositorio privado de imagens Docker na AWS.
# Aqui que o "docker push" envia a imagem buildada.
resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  # Scan automatico de vulnerabilidades nas imagens
  image_scanning_configuration {
    scan_on_push = true
  }
}

# Politica de ciclo de vida: mantem apenas as ultimas 5 imagens
# Sem isso, o ECR acumula imagens antigas e voce paga storage a toa
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Manter apenas as 5 imagens mais recentes"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ============================================================================
# 2. VPC — Virtual Private Cloud
# ============================================================================
# Rede privada isolada na AWS. Todos os recursos ficam dentro dela.
# CIDR 10.0.0.0/16 = ~65.000 IPs disponiveis
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

# ============================================================================
# 3. Subnets Publicas (2 AZs)
# ============================================================================
# Sub-redes dentro da VPC, cada uma em uma AZ diferente.
# "Publica" = recursos nela podem ter IP publico e acessar a internet.
# 2 AZs pra alta disponibilidade (o ALB exige no minimo 2).
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index) # 10.0.0.0/24, 10.0.1.0/24
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-${count.index}"
  }
}

# ============================================================================
# 4. Internet Gateway + Route Table
# ============================================================================
# Internet Gateway: conecta a VPC a internet.
# Sem isso, nada dentro da VPC consegue acessar (ou ser acessado pela) internet.
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

# Route Table: define que trafego para 0.0.0.0/0 (qualquer destino externo) vai pro IGW
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

# Associa a route table as subnets publicas
resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ============================================================================
# 5. Security Groups (Firewall)
# ============================================================================

# SG do ALB: aceita trafego HTTP (porta 80) de qualquer IP
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "Security group do ALB - permite HTTP da internet"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP da internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-alb-sg"
  }
}

# SG do ECS: so aceita trafego VINDO DO ALB (nao fica exposto direto na internet)
resource "aws_security_group" "ecs" {
  name        = "${var.project_name}-ecs-sg"
  description = "Security group do ECS - permite trafego apenas do ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Trafego do ALB"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ecs-sg"
  }
}

# ============================================================================
# 6. ALB — Application Load Balancer
# ============================================================================
# Distribui as requisicoes entre os containers do ECS.
# Se tiver 2 containers rodando, o ALB manda 50% pra cada.
resource "aws_lb" "app" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = {
    Name = "${var.project_name}-alb"
  }
}

# Target Group: grupo de alvos (containers) que o ALB roteia
resource "aws_lb_target_group" "app" {
  name        = "${var.project_name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip" # Fargate usa IPs, nao instancias EC2

  health_check {
    enabled             = true
    path                = var.health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }
}

# Listener: escuta na porta 80 e encaminha pro Target Group
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# ============================================================================
# 7. IAM Roles — Permissoes
# ============================================================================

# Execution Role: usado pelo ECS Agent pra gerenciar o container
# (puxar imagem do ECR, enviar logs pro CloudWatch)
resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Politica padrao da AWS para ECS (inclui ECR pull + CloudWatch logs)
resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Task Role: permissoes que o codigo DENTRO do container tem.
# Se precisar acessar S3, DynamoDB, etc, adiciona aqui.
resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# ============================================================================
# 8. CloudWatch Log Group — Logs centralizados
# ============================================================================
# Todos os logs do container (stdout/stderr) vao pra ca.
# Da pra ver tudo pelo console da AWS.
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 14

  tags = {
    Name = "${var.project_name}-logs"
  }
}

# ============================================================================
# 9. ECS Cluster
# ============================================================================
# Agrupamento logico de tasks/services.
# Com Fargate, nao precisa gerenciar servidores EC2 — a AWS cuida disso.
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ============================================================================
# 10. ECS Task Definition — Definicao do Container
# ============================================================================
# Descreve COMO o container deve rodar: imagem, CPU, RAM, portas, logs.
# Mesma ideia do docker-compose, mas pro ECS Fargate.
resource "aws_ecs_task_definition" "app" {
  family                   = var.project_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = var.project_name
      image     = "${aws_ecr_repository.app.repository_url}:latest"
      cpu       = var.cpu
      memory    = var.memory
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "LOG_LEVEL"
          value = "INFO"
        },
        {
          name  = "MODEL_PATH"
          value = "/app/models"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health')\" || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 15
      }
    }
  ])
}

# ============================================================================
# 11. ECS Service — Garante que os containers estejam rodando
# ============================================================================
# Mantem "desired_count" tasks SEMPRE rodando.
# Se um container morrer, o ECS sobe outro automaticamente.
resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  # Registra os containers no Target Group do ALB
  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.project_name
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.http]
}
