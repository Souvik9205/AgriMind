@echo off
REM AgriMind Docker Startup Script for Windows
REM This script starts the entire AgriMind stack with databases

setlocal enabledelayedexpansion

echo 🌱 Starting AgriMind with Docker Compose...
echo ==========================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ docker-compose is not installed. Please install Docker Compose.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📋 Creating .env file from template...
    copy .env.example .env
    echo ✅ Created .env file. You may want to customize it with your API keys.
)

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist data mkdir data
if not exist models mkdir models
if not exist uploads mkdir uploads
if not exist logs mkdir logs

REM Build and start services
echo 🚀 Building and starting services...
echo This may take a few minutes on the first run...

docker-compose up --build -d

echo.
echo ⏳ Waiting for services to be healthy...

REM Wait for PostgreSQL to be ready
echo 🗄️  Waiting for PostgreSQL...
:wait_postgres
docker-compose exec -T db pg_isready -U agrimind -d agrimind >nul 2>&1
if errorlevel 1 (
    echo|set /p="."
    timeout /t 2 /nobreak >nul
    goto wait_postgres
)
echo  ✅ PostgreSQL is ready!

REM Wait for Redis to be ready
echo 🔄 Waiting for Redis...
:wait_redis
docker-compose exec -T redis redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo|set /p="."
    timeout /t 1 /nobreak >nul
    goto wait_redis
)
echo  ✅ Redis is ready!

REM Wait for API to be ready
echo 🔌 Waiting for API server...
:wait_api
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo|set /p="."
    timeout /t 2 /nobreak >nul
    goto wait_api
)
echo  ✅ API server is ready!

echo.
echo 🎉 AgriMind is now running!
echo ==========================================
echo 📊 Services Status:
docker-compose ps

echo.
echo 🌐 Access Points:
echo    • API Server: http://localhost:8000
echo    • API Health: http://localhost:8000/health
echo    • API Docs: http://localhost:8000/docs
echo    • Database: localhost:5432 (agrimind/agrimind)
echo    • Redis: localhost:6379
echo.
echo 📋 Useful Commands:
echo    • View logs: docker-compose logs -f
echo    • Stop services: docker-compose down
echo    • Restart: docker-compose restart
echo.
echo 💡 For more information, see DOCKER_SETUP.md

pause