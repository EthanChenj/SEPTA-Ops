-- ==============================================================================
-- Migration 002: Least-Privilege Read-Only Role for Operations Copilot
-- ==============================================================================

DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles WHERE rolname = 'readonly_copilot'
   ) THEN
      CREATE ROLE readonly_copilot WITH LOGIN PASSWORD 'readonly_secret';
   END IF;
END
$do$;

-- Connect permissions
GRANT CONNECT ON DATABASE septa_ops TO readonly_copilot;

-- Schema usage
GRANT USAGE ON SCHEMA public TO readonly_copilot;

-- Read-only SELECT permissions on all existing tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_copilot;

-- Ensure future created tables also grant SELECT
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_copilot;

-- Revoke all mutation permissions explicitly
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA public FROM readonly_copilot;
REVOKE CREATE ON SCHEMA public FROM readonly_copilot;

-- Enforce safe guardrails on the role connection
ALTER ROLE readonly_copilot SET default_transaction_read_only = on;
ALTER ROLE readonly_copilot SET statement_timeout = '10s';
ALTER ROLE readonly_copilot SET idle_in_transaction_session_timeout = '10s';
