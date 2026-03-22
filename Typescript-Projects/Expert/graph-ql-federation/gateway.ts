// gateway.ts
import { ApolloGateway } from '@apollo/gateway';
import { ApolloServer } from 'apollo-server';

const gateway = new ApolloGateway({
    serviceList: [
        { name: 'accounts', url: 'http://localhost:4001' },
        { name: 'reviews', url: 'http://localhost:4002' }
    ]
});

const server = new ApolloServer({ gateway });
server.listen().then(({ url }) => console.log(`Gateway ready at ${url}`));
