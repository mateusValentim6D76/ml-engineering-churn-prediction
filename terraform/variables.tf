variable "project_name" {
  description = "Nome do projeto"
  type        = string
  default     = "churn-prediction"
}

variable "environment" {
  description = "Ambiente de deploy"
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "Região da AWS"
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "CIDR block da VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "container_port" {
  description = "Porta do container"
  type        = number
  default     = 8000
}

variable "cpu" {
  description = "CPU para a task ECS (256 = 0.25 vCPU)"
  type        = number
  default     = 256
}

variable "memory" {
  description = "Memória para a task ECS (em MB)"
  type        = number
  default     = 512
}

variable "desired_count" {
  description = "Número de instâncias do container"
  type        = number
  default     = 1
}

variable "health_check_path" {
  description = "Path do health check"
  type        = string
  default     = "/health"
}
