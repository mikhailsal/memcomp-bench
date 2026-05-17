import { Routes, Route } from 'react-router-dom'
import { ManifestProvider } from './lib/manifest'
import Layout from './components/Layout'
import ConversationList from './pages/ConversationList'
import ConversationView from './pages/ConversationView'
import About from './pages/About'

export default function App() {
  return (
    <ManifestProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<ConversationList />} />
          <Route path="/conversation/:id" element={<ConversationView />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Layout>
    </ManifestProvider>
  )
}
