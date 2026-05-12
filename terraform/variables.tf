
# ── Projeto ──────────────────────────────────────────────────────────────────

variable "project_name" {
  description = "Nome do projeto (usado como prefixo em todos os recursos)"
  type        = string
  default     = "churn-prediction"
}

variable "environment" {
  description = "Ambiente de deploy (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# ── AWS ──────────────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "Região da AWS onde os recursos serão criados"
  type        = string
  default     = "us-east-1"
}

# ── Rede (VPC) ───────────────────────────────────────────────────────────────

variable "vpc_cidr" {
  description = "CIDR block da VPC (range de IPs da rede privada)"
  type        = string
  default     = "10.0.0.0/16"
}

# ── ECS (Container) ─────────────────────────────────────────────────────────

variable "container_port" {
  description = "Porta que o container expõe (deve bater com o EXPOSE do Dockerfile)"
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "CPU para a task ECS (em unidades: 256 = 0.25 vCPU)"
  type        = number
  default     = 256
}

variable "memory" {
  description = "Memória para a task ECS (em MB)"
  type        = number
  default     = 512
}

variable "desired_count" {
  description = "Número de instâncias (tasks) do container rodando simultaneamente"
  type        = number
  default     = 1
}

variable "health_check_path" {
  description = "Path do health check usado pelo ALB"
  type        = string
  default     = "/health"
}
