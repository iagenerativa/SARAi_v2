"""
skills/sql/__init__.py - SQL Skill Especializado v2.12 Phoenix

PROPÓSITO:
- Generación de queries SQL optimizadas
- Modelo: CodeLlama-7B-Instruct-SQL (Q4_K_M, ~4GB)
- Contexto: 512 tokens (suficiente para schemas pequeños)

EJEMPLO DE USO:
  # En el host (via gRPC client)
  from core.model_pool import get_skill_client
  
  skill = get_skill_client("sql")
  result = skill.Execute(
      query="Genera un SELECT para obtener usuarios activos de la tabla users",
      max_tokens=128
  )
  print(result.response)

MODELO REQUERIDO:
  # Descargar CodeLlama-SQL GGUF
  huggingface-cli download TheBloke/CodeLlama-7B-Instruct-SQL-GGUF \\
    codellama-7b-instruct-sql.Q4_K_M.gguf \\
    --local-dir models/skills/
  
  # Mover a ruta esperada
  cp models/skills/codellama-7b-instruct-sql.Q4_K_M.gguf models/sql.gguf
"""

__version__ = "2.12.0"
__skill_name__ = "sql"
__model_file__ = "sql.gguf"  # Esperado en /app/models/sql.gguf dentro del contenedor

# Metadata del skill
SKILL_CONFIG = {
    "name": "sql",
    "description": "Generación de queries SQL optimizadas",
    "model": "CodeLlama-7B-Instruct-SQL (Q4_K_M)",
    "context_length": 512,
    "max_tokens": 256,
    "temperature": 0.2,  # Baja temperatura para precisión
    "specialization": ["SQL generation", "Query optimization", "Schema design"]
}

def get_system_prompt() -> str:
    """
    System prompt optimizado para SQL
    
    Returns:
        str: Prompt que maximiza precisión en SQL
    """
    return """You are a SQL expert assistant. Generate precise, optimized SQL queries.

Rules:
1. Use standard SQL syntax (PostgreSQL compatible)
2. Always use table aliases
3. Prefer JOINs over subqueries when possible
4. Include LIMIT clauses for safety
5. Add comments for complex queries

Output format: SQL code only, no explanations unless requested."""

def validate_sql_syntax(query: str) -> tuple[bool, str]:
    """
    Validación básica de sintaxis SQL
    
    Args:
        query: Query SQL generada
    
    Returns:
        (is_valid, error_message)
    """
    # Validaciones básicas
    query_upper = query.upper()
    
    # Debe contener al menos un comando SQL
    sql_commands = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER"]
    if not any(cmd in query_upper for cmd in sql_commands):
        return False, "No SQL command detected"
    
    # Balanceo de paréntesis
    if query.count('(') != query.count(')'):
        return False, "Unbalanced parentheses"
    
    # Comillas balanceadas
    if query.count("'") % 2 != 0:
        return False, "Unbalanced quotes"
    
    return True, ""

# Example usage (para testing)
if __name__ == "__main__":
    print(f"SQL Skill v{__version__}")
    print(f"Config: {SKILL_CONFIG}")
    print(f"\nSystem Prompt:\n{get_system_prompt()}")
