output "api_url" {
  description = "URL da API"
  value       = "http://${aws_lb.app.dns_name}"
}

output "ecr_repository_url" {
  description = "URL do ECR"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "Nome do cluster ECS"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Nome do service ECS"
  value       = aws_ecs_service.app.name
}

output "cloudwatch_log_group" {
  description = "Log group no CloudWatch"
  value       = aws_cloudwatch_log_group.app.name
}
