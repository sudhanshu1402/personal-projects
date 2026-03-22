// pages/index.tsx
import { GetStaticProps } from 'next';

interface Product {
    id: number;
    name: string;
}

export const getStaticProps: GetStaticProps = async () => {
    // Fetch data
    return { props: { products: [{ id: 1, name: "Item 1" }] } };
}

export default function Home({ products }: { products: Product[] }) {
    return <ul>{products.map(p => <li key={p.id}>{p.name}</li>)}</ul>;
}
