// main.ts
import { NestFactory } from '@nestjs/core';
import { Module, Controller, Get } from '@nestjs/common';

@Controller('cats')
class CatsController {
    @Get()
    findAll(): string {
        return 'This action returns all cats';
    }
}

@Module({
    controllers: [CatsController],
})
class AppModule { }

async function bootstrap() {
    const app = await NestFactory.create(AppModule);
    await app.listen(3000);
}
bootstrap();
